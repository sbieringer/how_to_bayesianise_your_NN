import torch
from copy import deepcopy as dc
from numpy import pi, sqrt, log
from numpy.random import rand

def diff(params1, params2):
    diff = params1 - params2
    return diff.detach()

def log_normal(diff_eval, sigma, diff_adam=None, sigma_adam_dir=0, verbose = False):
    with torch.no_grad():
        d = len(diff_eval)
        len2_diff_eval, len2_diff_adam = torch.sum(diff_eval**2), torch.sum(diff_adam**2)
        
        difference_prod = torch.inner(diff_eval, diff_adam) if diff_adam is not None else 0
        directional_factor = 1/(sigma**4/sigma_adam_dir**2 + sigma**2*len2_diff_adam) if diff_adam is not None else 0
        log_norm = -d*log(sigma) if diff_adam is None else -d*log(sigma)-torch.log((sigma_adam_dir/sigma)**2*len2_diff_adam+1)  #should include a 2\pi**d/2 but that cancels anyhow 
        
        log_normal_out = log_norm-0.5*(len2_diff_eval/sigma**2 - directional_factor*difference_prod**2)
        return log_normal_out
    

def accept_prop(temp, old_loss, prop_loss, state_diff, old_state, new_state, prop_state, next_state, sigma, sigma_adam_dir = 0, verbose = False):
    with torch.no_grad():
        diff_old, diff_prop = diff(old_state, next_state), diff(prop_state, new_state)        

        prop_prob_old = log_normal(diff_old, sigma, state_diff, sigma_adam_dir, verbose) 
        prop_prob_prop = log_normal(diff_prop, sigma, state_diff, sigma_adam_dir, verbose)
        
        val = torch.exp(temp*(old_loss-prop_loss) + prop_prob_old - prop_prob_prop)# + log_norms)

        val_ret = torch.nan_to_num(val, 0, posinf = 1, neginf = 0)
        val_ret = torch.clip(val_ret, None, 1e6)
    
    if verbose:
        print(f'acc_prob: {val_ret} \nloss old: {old_loss}, prop loss: {prop_loss} \nacceptance on loss: {torch.exp(temp*(old_loss-prop_loss))} \nacceptance on proposal distribution {torch.exp(prop_prob_old - prop_prob_prop)}')

    return val_ret, val, prop_prob_old-prop_prob_prop

class MCMC_by_bp():
    def __init__(self, model, optimizer, temperature, sigma=1.0, sigma_adam_dir = None, **kwargs):
        '''
        class implementing the cycSGMH algorithm. 
        Can be used similar to torchch.optimizer.Optimizer instances, but loss needs to be defined for fixed newtork weights.

        args: model - torch.nn.Module: NN model to optimize with optimizer, needed to give acces to model.stat_dict() 
              optimizer - torch.optimizer.Optimizer: instance of the torch.optimizer class already initialized with the parameters in question and the hyperparameters of choice
              temperature - FLOAT: temperature parameter. Higher parameter makes the Gibss-posterior sharper and the uncertainty smaller.
              sigma - FLOAT: if float standard deviation of the noise added to the update step (normailized by 1/\sqrt(n_weights)), default = 1.0
              sigma_adam_dir - FLOAT: if float standard deviation of the noise added to the update step in step direction (normailized by 1/\sqrt(n_weights)), default = None
                                                
        '''
        self.model = model
        self.opt = optimizer
        self.temp = temperature
        self.sigma = sigma
        self.sigma_adam_dir = sigma_adam_dir
        self.n_weights = torch.sum(torch.Tensor([x.numel() for x in self.model.parameters()])).item()
        
        self.theta_k = None
        self.start = True

                
    def step(self, loss, **kwargs):
        '''
        single step in the cycSGMH algorithm
        
        args: loss - callable: Returns the value of the loss for the current network weights (!!!)
            
        kwargs: verbose - BOOL: verbose output, default = False
                extended_doc_dict - BOOL: whether to return all important values of the step in a dict, default = False

                MH - BOOL: whethter to do a MH step, defalut = True
                full_loss - False, Callable: if callable full_loss() is called to determine the loss in MH step, if False loss() is called, default = False
                
                gamma_sigma_decay - FLOAT: exponential decay of sigma, default = None
                sigma_min - FLOAT: mimimim of exponential decay, default = 0
                sigma_factor - FLOAT: factor multiplied with sigma to get the proposal distribution standard deviation, default = 1
                sigma_adam_dir - FLOAT: resets the initialized sigma_adam_dir, in case you want to do some scheduling, default = None            
        '''
        
        #kwargs
        verbose = kwargs.get('verbose', False)
        full_loss = kwargs.get('full_loss', False)
        use_full_loss = hasattr(full_loss, '__call__')

        extended_doc_dict = kwargs.get('extended_doc_dict', False)
        
        gamma_sigma_decay = kwargs.get('gamma_sigma_decay')
        sigma_min = kwargs.get('sigma_min', 0)
        sigma_factor = kwargs.get('sigma_factor', 1)
        sigma_adam_dir = kwargs.get('sigma_adam_dir') if kwargs.get('sigma_adam_dir') is not None else self.sigma_adam_dir
        
        MH = kwargs.get('MH', True)

        #initialize the doc dict
        if extended_doc_dict:
            with torch.no_grad():
                doc_dict = {'loss_diff': torch.Tensor([torch.nan]).to(self.model.device), 
                            'old_loss': torch.Tensor([torch.nan]).to(self.model.device), 
                            'prop_loss': torch.Tensor([torch.nan]).to(self.model.device), 
                            'prob_diff': torch.Tensor([torch.nan]).to(self.model.device), 
                            'full_train_loss': torch.Tensor([torch.nan]).to(self.model.device), 
                        }
        else:
            doc_dict = {}
            
        self.opt.zero_grad()

        #set old state
        if self.start:
            self.theta_k = self.get_params() 

        #calculate \tilda\theta^{(k+1)}
        L_theta_k = loss() 
        L_theta_k.backward()
        L_theta_k_acc = full_loss(self.model) if use_full_loss else L_theta_k
        self.opt.step()
        tilde_theta_k1 = self.get_params()

        #calulate u_k for the backwards direction
        with torch.no_grad():
            u_k = diff(self.theta_k, tilde_theta_k1)
                    
        if extended_doc_dict:
            if use_full_loss:
                doc_dict['full_train_loss'] = L_theta_k

        #normalize sigma to the number of network weights
        if gamma_sigma_decay is not None and self.sigma > sigma_min:
            self.sigma = self.sigma*gamma_sigma_decay
        sigma = dc(self.sigma)
        sigma_norm_factor = sqrt(1/(self.n_weights)) 
        sigma *= sigma_norm_factor*sigma_factor
        sigma_adam_dir = sigma_adam_dir*sigma_norm_factor if sigma_adam_dir is not None else sigma_adam_dir
        
        #calculate \tau^{(k)}
        if sigma_adam_dir is not None:
            sigma_item = sigma
            sampled_sigma = torch.normal(0, sigma_item, tilde_theta_k1.shape, device = self.model.device)
            sampled_sigma_adam = torch.normal(0, sigma_adam_dir, tilde_theta_k1.shape, device = self.model.device)
            tau_k = sampled_sigma + tilde_theta_k1 - u_k*sampled_sigma_adam
        else:
            tau_k = torch.normal(0, sigma_item, tilde_theta_k1.shape, device = self.model.device) + tilde_theta_k1

        #do the MH correction
        if MH:
            self.set_params(tau_k)

            L_tau_k = loss()
            L_tau_k_acc = full_loss(self.model) if use_full_loss else L_tau_k

            #calculate \tilde\tau^{(k+1)}
            tilde_tau_k1 = tau_k-u_k

            a, a_full,prob_diff_doc = accept_prop(self.temp, L_theta_k_acc, L_tau_k_acc, u_k,
                                                     self.theta_k, tilde_theta_k1, tau_k, tilde_tau_k1, 
                                                     sigma, sigma_adam_dir = sigma_adam_dir,
                                                     verbose = verbose) 
            if extended_doc_dict:
                doc_dict['prob_diff'] = prob_diff_doc
                doc_dict['loss_diff'] = self.temp*(L_theta_k_acc-L_tau_k_acc)
                doc_dict['old_loss']  = L_theta_k_acc
                doc_dict['prop_loss'] = L_tau_k_acc

            #accept or reject the proposed state
            u = rand(1)
            if verbose:
                print(f"u {u} -> accepted {u <= a.cpu().numpy()}")
            if u <= a.cpu().numpy():
                self.theta_k = tau_k 
                accepted = True
            else:
                #and set the theta_k for the next iteration
                self.set_params(self.theta_k)
                accepted = False

        # or dont do MH
        else:
            self.theta_k = tau_k 
            self.set_params(self.theta_k)
            a, a_full, accepted = torch.Tensor([1]), torch.Tensor([1]), True
    
        return self.theta_k, a_full, accepted, sigma, doc_dict

    def get_params(self):
        return torch.nn.utils.parameters_to_vector(self.model.parameters()).detach()

    def set_params(self, out):
        torch.nn.utils.vector_to_parameters(out, self.model.parameters())
            
    def print_lr(self):
        return self.opt.param_groups[0]['lr']
        
    def zero_grad(self):
        self.opt.zero_grad()
        
    def state_dict(self):
        return self.opt.state_dict()
    
    def load_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict)