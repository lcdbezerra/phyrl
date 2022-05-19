import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd.functional import jvp
import numpy as np

from utils import init_params, init_sindy
    
def get_act_from_txt(txt):
        if txt == "relu":
            return nn.ReLU
        elif txt == "leaky_relu":
            return nn.LeakyReLU
        elif txt == "sigmoid":
            return nn.Sigmoid
        else:
            raise ValueError("Activation function not supported")
    
class MLP(nn.Module):
    def __init__(self, arch, act="sigmoid"):
        super().__init__()
        self.arch = arch
        self.act = get_act_from_txt(act)
        
        if len(arch)==2:
            self.net = nn.Linear(arch[0], arch[1])
        elif len(arch)>2:
            lst = []
            for i in range(len(arch)-2): # last layer has no activation
                lst += [
                    nn.Linear(arch[i], arch[i+1]),
                    self.act()
                ]
            lst += [ nn.Linear(arch[-2], arch[-1]) ]
            self.net = nn.Sequential(*lst)
        else:
            raise ValueError("Bad MLP architecture")
    
    def forward(self, x):
        return self.net(x)
    
    def __getitem__(self, key):
        # return self.net[key]
        return self.net.__getitem__(key)
    def __len__(self):
        # return len(self.net)
        return self.net.__len__()

class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        assert(config.model_order in [1,2])
        self.model_order = config.model_order
        self.control_size = config.control_size
        self.control_only = config.control_only
        self.bs = config.batch_size
        self.phy_constraint = config.phy_constraint
        self.loss_fn = nn.MSELoss()
        self.done_fn = nn.CrossEntropyLoss()
        
        self.enc_shape = config.enc_shape
        # self.enc_act = config.enc_activation
        if config.dec_shape is None:
            self.dec_shape = config.enc_shape[::-1]
        else:
            self.dec_shape = config.dec_shape
        # self.dec_act = config.dec_activation
        
        self.in_sz = config.in_sz
        self.dim_sz = config.dim_sz
        
        self.act = get_act_from_txt(config.act)
        
        if config.batch_norm:
            self.bn = nn.BatchNorm1d(self.in_sz)
        else:
            self.bn = lambda x: x
        
        # Build encoder
        enc_arch = (self.in_sz, *self.enc_shape, self.dim_sz)
        self.encoder = MLP(enc_arch, act=config.act)
        self.encoder.apply(init_params)        
        
        # Build decoder
        dec_arch = (self.dim_sz, *self.dec_shape, self.in_sz)
        self.decoder = MLP(dec_arch, act=config.act)
        self.decoder.apply(init_params)
        
        # Initialize SINDy
        if config.phy_constraint is None:
            self.sindy = SINDy(config)
        elif config.phy_constraint == 1:
            self.sindy = Constrained1(config)
        elif config.phy_constraint == 2:
            self.sindy = Constrained2(config)
        elif config.phy_constraint == 3:
            self.sindy = Constrained3(config)
        elif config.phy_constraint == 4:
            self.sindy = ConstrainedSINDy(config)
        else:
            raise ValueError(f"Bad physics model, {config.phy_constraint}")

        if config.phy_constraint is None:
            self.sindy_sparsity_loss = lambda: torch.mean(torch.norm(self.sindy.coeff_layer.weight, p=1))
        else:
            self.sindy_sparsity_loss = lambda: torch.zeros((1,)).to(self.config.device)
        
        # Initialize reward model
        d = self.dim_sz if self.config.model_order==1 else 2*self.dim_sz
        if self.control_size is not None:
            d += self.control_size
        rwd_arch = (d, *config.rwd_shape, 1)
        self.rwd_model = MLP(rwd_arch, act=config.act)
        
        # Initialize done model
        done_arch = (d, *config.done_shape, 2)
        self.done_model = MLP(done_arch, act=config.act)
        self.done_fn = nn.CrossEntropyLoss(weight=torch.Tensor([.9,.1]))

    def prepareDataForSindy(self, z,u):
        if self.phy_constraint is None:
            raise ValueError("Prepare data for SINDy should be used for constrained models only.")
        elif self.phy_constraint == 1:
            return self.sindy(u)
        elif self.phy_constraint == 2:
            return self.sindy(u)
        elif self.phy_constraint == 3:
            return self.sindy(z,u)
        elif self.phy_constraint == 4:
            return self.sindy(z,u)
        else:
            raise ValueError(f"Bad physics model, {config.phy_constraint}")

        
    def forward(self,x):
        raise NotImplementedError("Forward function")
        # FIX!!!!!!!!!!!!!!!!
        # 2ND ORDER
        x = x.flatten(start_dim=1)
        x = self.bn(x)
        z = self.encoder(x)
        if self.control_only:
            theta = u
        elif self.control_size is None:
            theta = z
        else:
            theta = torch.cat((z,u), dim=-1)
        ddz = self.sindy(z)
        xh = self.decoder(z)
        
        return z,ddz,xh
            
    def loss(self, data):
        
        if self.model_order == 1:
            raise NotImplementedError("1st-order model out of date")
            return self.loss_order1(data)
        elif self.model_order == 2:
            return self.loss_order2(data)
        else:
            raise ValueError("Model order (%d) not supported" % self.model_order)
    
    def s(self,x):
        return self.act()(x)
    
    def ds(self,x):
        return self.s(x)*(1-self.s(x))
    
    def dds(self,x):
        return -self.s(x)*(1-self.s(x))*(1-2*self.s(x))
    
    def dz_dt_hardcoded(self, x, dx):
        # Assume activation function is sigmoid
        assert(isinstance(self.act(), torch.nn.modules.activation.Sigmoid))
        
        dz = dx.mm(self.encoder[0].weight.T)
        z  = self.encoder[0](x)
        for i in range(2,len(self.encoder),2):
            dz = ( self.ds(z)*dz ).mm(self.encoder[i].weight.T)
            z  = self.s(z)
            z  = self.encoder[i](z)
        return z,dz
    
    def dxhat_dt_hardcoded(self, z, dz):
        dx = dz.mm(self.decoder[0].weight.T)
        x  = self.decoder[0](z)
        for i in range(2,len(self.decoder),2):
            dx = ( self.ds(x)*dx ).mm(self.decoder[i].weight.T)
            x  = self.s(x)
            x  = self.decoder[i](x)
        return x,dx
    
    def loss_order1(self, data):
        x   = data[0].to(self.config.device)
        dx  = data[1].to(self.config.device)
        u   = data[2].to(self.config.device)
        rwd = data[3].to(self.config.device)
        lmb1, lmb2, lmb3, lmb4, lmb5 = self.config.lmb
        
        # JVP: Jacobian-vector product
        # Compute dz_dt
        z, dz_dt = jvp(self.encoder, x, v=dx, create_graph=True)
        # Compute dx^_dt.
        if self.control_only:
            theta = u
        elif self.control_size is None:
            theta = z
        else:
            theta = torch.cat((z,u), dim=-1)
        
        dz_dt_sindy = self.sindy(theta)
        xhat, dxhat_dt = jvp(self.decoder, z, v=dz_dt_sindy, create_graph=True)
        
#         z1, dz_dt1 = self.dz_dt_hardcoded(x,dx)
        # assert torch.max(torch.abs(z1-z)) <= 1e-3, "1 Autograd computation difference"
        # assert torch.max(torch.abs(dz_dt1-dz_dt)) <= 1e-3, "2 Autograd computation difference"
        
#         xhat1, dxhat_dt1 = self.dxhat_dt_hardcoded(z,dz_dt_sindy)
#         assert torch.max(torch.abs(xhat1-xhat)) <= 1e-3, "3 Autograd computation difference"
#         assert torch.max(torch.abs(dxhat_dt1-dxhat_dt)) <= 1e-3, "4 Autograd computation difference"

        if (self.control_size is None) and (not self.control_only):
            theta = z
        else:
            theta = torch.cat((z,u), dim=-1)
        pred_rwd = self.rwd_model(theta)
        
        losses = (self.loss_fn(pred_rwd, rwd),
                  self.loss_fn(x, xhat), 
                  self.loss_fn(dx, dxhat_dt), 
                  self.loss_fn(dz_dt, dz_dt_sindy),
                  torch.mean(torch.norm(self.sindy.coeff_layer.weight, p=1)),
         )

        loss = lmb1*self.loss_fn(x, xhat) + \
                lmb2*self.loss_fn(dx, dxhat_dt) + \
                lmb3*self.loss_fn(dz_dt, dz_dt_sindy) + \
                lmb4*torch.mean(torch.norm(self.sindy.coeff_layer.weight, p=1)) + \
                lmb5*self.loss_fn(pred_rwd, rwd)
        return loss, losses

    ##############################################
    # SECOND-ORDER
    
    def forward_order2(self, data):
        x  = data[0].to(self.config.device).flatten(start_dim=1)
        dx = data[1].to(self.config.device).flatten(start_dim=1)
        u  = data[2].to(self.config.device)

        x = x.flatten(start_dim=1)
        x = self.bn(x)
        # Compute dz_dt
        z, dz_dt = jvp(self.encoder, x, v=dx, create_graph=True)
        xhat = self.decoder(z)
        # SINDy
        if self.control_only:
            theta = u
        elif self.control_size is None:
            theta = torch.cat((z,dz_dt), dim=-1)
        else:
            theta = torch.cat((z,dz_dt,u), dim=-1)
        d2z_dt2_sindy = self.sindy(theta)
        xh = self.decoder(z)
        
        if (self.control_size is None) and (not self.control_only):
            theta = torch.cat((z,dz_dt), dim=-1)
        else:
            theta = torch.cat((z,dz_dt,u), dim=-1)
        rwd  = self.rwd_model(theta)
        done = self.done_model(theta)
        done = done[0] >= done[1]

        return d2z_dt2_sindy, rwd, done, xh    
    
    def d2z_dt2_hardcoded(self, x, dx, ddx):
        # Assume activation function is sigmoid
        assert(isinstance(self.act(), torch.nn.modules.activation.Sigmoid))
        
        ddz = ddx.mm(self.encoder[0].weight.T)
        dz  = dx.mm(self.encoder[0].weight.T)
        z   = self.encoder[0](x)
        for i in range(2,len(self.encoder),2):
            ddz = ( self.dds(z)*dz**2 + self.ds(z)*ddz ).mm(self.encoder[i].weight.T)
            dz = ( self.ds(z)*dz ).mm(self.encoder[i].weight.T)
            z  = self.s(z)
            z  = self.encoder[i](z)
        return z,dz,ddz
    
    def d2xhat_dt2_hardcoded(self, x, dx, ddx):
        # same notation as function above, but using decoder instead
                
        ddz = ddx.mm(self.decoder[0].weight.T)
        dz  = dx.mm(self.decoder[0].weight.T)
        z   = self.decoder[0](x)
        for i in range(2,len(self.decoder),2):
            ddz = ( self.dds(z)*dz**2 + self.ds(z)*ddz ).mm(self.decoder[i].weight.T)
            dz = ( self.ds(z)*dz ).mm(self.decoder[i].weight.T)
            z  = self.s(z)
            z  = self.decoder[i](z)
        return z,dz,ddz
    
    def get_d2z_from_z(self, data):
        z     = data[0].to(self.config.device)
        dz_dt = data[1].to(self.config.device)
        u     = data[2].to(self.config.device)
        
        # if self.control_only:
        #     theta = u
        # elif self.control_size is None:
        #     theta = torch.cat((z,dz_dt), dim=-1)
        # else:
        #     theta = torch.cat((z,dz_dt,u), dim=-1)

        # SINDy
        if self.phy_constraint is None:
            if self.control_only:
                theta = u
            elif self.control_size is None:
                theta = torch.cat((z,dz_dt), dim=-1)
            else:
                theta = torch.cat((z,dz_dt,u), dim=-1)
            d2z_dt2_sindy = self.sindy(theta)
        else:
            d2z_dt2_sindy = self.prepareDataForSindy(z,u)

        return d2z_dt2_sindy
        
    def loss_order2(self, data):
        x    = data[0].to(self.config.device).flatten(start_dim=1)
        dx   = data[1].to(self.config.device).flatten(start_dim=1)
        ddx  = data[2].to(self.config.device).flatten(start_dim=1)
        u    = data[3].to(self.config.device).float()
        rwd  = data[4].to(self.config.device).float()
        done = data[5].to(self.config.device).float()
        # lmb  = torch.Tensor(self.config.lmb).to(self.config.device)
        lmb  = self.config.lmb
        
        # assert len(lmb)==6, "Bad loss weights"
        
        # Batch normalization
        x = self.bn(x)

        # JVP: Jacobian-vector product
        # Compute dz_dt
        z, dz_dt = jvp(self.encoder, x, v=dx, create_graph=True)
        xhat = self.decoder(z)
        # SINDy
        if self.phy_constraint is None:
            if self.control_only:
                theta = u
            elif self.control_size is None:
                theta = torch.cat((z,dz_dt), dim=-1)
            else:
                theta = torch.cat((z,dz_dt,u), dim=-1)
            d2z_dt2_sindy = self.sindy(theta)
        else:
            d2z_dt2_sindy = self.prepareDataForSindy(z,u)


        z1, dz_dt1, d2z_dt2 = self.d2z_dt2_hardcoded(x, dx, ddx)
        xhat, _, d2xhat_dt2 = self.d2xhat_dt2_hardcoded(z, dz_dt, d2z_dt2_sindy)
        
        
        if self.control_size is None:
            theta = torch.cat((z,dz_dt), dim=-1)
        else:
            theta = torch.cat((z,dz_dt,u), dim=-1)
        pred_rwd  = self.rwd_model(theta)
        pred_done = self.done_model(theta)

        losses = (
            self.loss_fn(pred_rwd, rwd),
            self.done_fn(pred_done, done),
            self.loss_fn(x, xhat),
            self.loss_fn(ddx, d2xhat_dt2),
            self.loss_fn(d2z_dt2, d2z_dt2_sindy),
            self.sindy_sparsity_loss(),
        )

        loss = sum([lmb[i]*losses[i] for i in range(len(losses))])

        # loss = lmb1*self.loss_fn(x, xhat) + \
        #         lmb2*self.loss_fn(ddx, d2xhat_dt2) + \
        #         lmb3*self.loss_fn(d2z_dt2, d2z_dt2_sindy) + \
        #         lmb4*torch.mean(torch.norm(self.sindy.coeff_layer.weight, p=1)) + \
        #         lmb5*self.loss_fn(pred_rwd, rwd)
        return loss, losses
    
    def losses_names(self):
        names = (
            "Reward prediction",
            "End of episode prediction",
            "Reconstruction",
            "Reconstruction time derivative",
            "Latent representation time derivative",
            "Sparsity",
        )
        return names

class Constrained1(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.d = config.dim_sz
        assert config.dim_sz == 2, "Expected z dim. = 2"
    
    def forward(self,u):
        return torch.stack((u[:,1], u[:,3]-u[:,2]), dim=-1)

class Constrained2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.d = config.dim_sz
        assert config.dim_sz == 2, "Expected z dim. = 2"
        self.a1 = nn.Parameter(torch.ones((1,)).to(self.config.device))
        self.a2 = nn.Parameter(torch.ones((1,)).to(self.config.device))
    
    def forward(self,u):
        return torch.stack( ( self.a1*u[:,1], self.a2*(u[:,3]-u[:,2]) ), dim=-1)

class Constrained3(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.d = config.dim_sz
        assert config.dim_sz == 3, "Expected z dim. = 3"
        self.a = nn.Parameter(torch.ones((1,)).to(self.config.device))
    
    def forward(self,z,u):
        theta = z[:,-1]
        return torch.stack((
            -torch.sin(theta)*u[:,1] + torch.cos(theta)*(u[:,3]-u[:,2]), 
            torch.cos(theta)*u[:,1] + torch.sin(theta)*(u[:,3]-u[:,2]), 
            self.a*(u[:,2]-u[:,3]),
        ), dim=-1)

class ConstrainedSINDy(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.d = config.dim_sz
        assert config.dim_sz == 3, "Expected z dim. = 3"
        self.lib_sz = 1+4+4+4 # bias, u, sin*u, cos*u

        # Coefficients (initialized as ones)
        self.coeff_layer = nn.Linear(self.lib_sz, self.d, bias=False).to(self.config.device)
        self.coeff_layer.apply(init_sindy)

    def library(self, z, u):
        d = 1+4+4+4 # bias, u, sin*u, cos*u
        assert 3==z.shape[1], f"Incompatible sizes in SINDy: z {z.shape[1]}, expected {3}"
        assert 4==u.shape[1], f"Incompatible sizes in SINDy: u {u.shape[1]}, expected {4}"

        nt = z.shape[0]
        lib = torch.zeros(nt, d).to(self.config.device)
        theta = z[:,-1].view(-1,1)

        # Constant
        lib[:,0] = 1
        lib_i = 1
        # Linear in u
        lib[:,lib_i:lib_i+4] = u
        lib_i += 4
        # Sin(theta)*u
        lib[:,lib_i:lib_i+4] = torch.sin(theta)*u
        lib_i += 4
        # Cos(theta)*u
        lib[:,lib_i:lib_i+4] = torch.cos(theta)*u
        lib_i += 4
        return lib

    def forward(self, z,u):
        return self.coeff_layer(self.library(z,u))



class SINDy(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.d = config.dim_sz
        # self.d = config.dim_sz if config.model_order==1 else 2*config.dim_sz
        assert(config.poly_order <= 5) # maximum order supported
        self.poly_order = config.poly_order
        self.include_sine = config.include_sine
        self.include_cos = config.include_cos
        self.control_size = config.control_size
        self.control_only = config.control_only
        
        self.lib_size = self.compute_lib_size()
        
        # Coefficients (initialized as ones)
        self.coeff_layer = nn.Linear(self.lib_size, self.d, bias=False)
        self.coeff_layer.apply(init_sindy)
        
    def compute_lib_size(self):
        if self.control_only:
            d = self.control_size
        else:
            d = self.d if self.config.model_order==1 else 2*self.d
            if self.control_size is not None:
                d += self.control_size
        
        sz = 1 # The ones function
        sz += d # Linear in z
        
        if self.poly_order >= 2:
            sz += sum([1 for i in range(d) for j in range(i,d)]) # Quadratic in z
        if self.poly_order >= 3:
            # 3rd-order poly
            sz += sum([1 for i in range(d) for j in range(i,d) for k in range(j,d)]) 
        if self.poly_order >= 4:
            # 4th-order poly
            sz += sum([1 for i in range(d) for j in range(i,d) for k in range(j,d) for p in range(k,d)]) 
        if self.poly_order >= 5:
            # 5th-order poly
            sz += sum([1 for i in range(d) for j in range(i,d) for k in range(j,d) for p in range(k,d) for q in range(p,d)]) 
        
        if self.include_sine:
            sz += d # Sine(z), Sine(u)
        if self.include_cos:
            sz += d # Sine(z), Sine(u)
        return sz
    
    def lib_description(self):
        # Returns array of strings describing each function
        # of the library
        lst = ["1"]
        raise NotImplementedError("Lib description not implemented")
        
    def poly_library(self, z):
        if self.control_only:
            d = self.control_size
        else:
            d = self.d if self.config.model_order==1 else 2*self.d
            if self.control_size is not None:
                d += self.control_size
        
        assert(d==z.shape[1])
        nt = z.shape[0]
        lib = torch.zeros(nt, self.lib_size).to(self.config.device)
        
        lib[:,0] = 1
        lib_i = 1
        
        # Linear in z
        lib[:,lib_i:lib_i+d] = z
        lib_i += d
        # Quadratic in z
        if self.poly_order >= 2:
            for i in range(d):
                for j in range(i,d):
                    lib[:,lib_i] = z[:,i]*z[:,j]
                    lib_i += 1
        # 3rd-order poly
        if self.poly_order >= 3:
            for i in range(d):
                for j in range(i,d):
                    for k in range(j,d):
                        lib[:,lib_i] = z[:,i]*z[:,j]*z[:,k]
                        lib_i += 1
        # 4th-order poly
        if self.poly_order >= 4:
            for i in range(d):
                for j in range(i,d):
                    for k in range(j,d):
                        for p in range(k,d):
                            lib[:,lib_i] = z[:,i]*z[:,j]*z[:,k]*z[:,p]
                            lib_i += 1
        # 5th-order poly
        if self.poly_order >= 5:
            for i in range(d):
                for j in range(i,d):
                    for k in range(j,d):
                        for p in range(k,d):
                            for q in range(p,d):
                                lib[:,lib_i] = z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q]
                                lib_i += 1
        # Sine(z)
        if self.include_sine:
            for i in range(d):
                lib[:,lib_i] = torch.sin(z[:,i])
                lib_i += 1
        # Cos(z)
        if self.include_cos:
            for i in range(d):
                lib[:,lib_i] = torch.cos(z[:,i])
                lib_i += 1
                
        assert(lib_i == self.lib_size)
        return lib
    
    def forward(self, z):
        return self.coeff_layer(self.poly_library(z))