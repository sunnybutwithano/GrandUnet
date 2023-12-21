from base_classes import ODEblock
import torch
from utils import get_rw_adj, gcn_norm_fill_val
# from run_GNN import Print


class ConstantODEblock(ODEblock):
  def __init__(self, odefunc, regularization_fns, opt, data, device, t=torch.tensor([0, 1])):
    super(ConstantODEblock, self).__init__(odefunc, regularization_fns, opt, data, device, t)
    self.opt = opt

    self.aug_dim = 2 if opt['augment'] else 1
    self.odefunc = odefunc(self.aug_dim * opt['hidden_dim'], self.aug_dim * opt['hidden_dim'], opt, data, device)
    if opt['data_norm'] == 'rw':
      
      edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                                                   fill_value=opt['self_loop_weight'],
                                                                   num_nodes=data.num_nodes,
                                                                   dtype=data.x.dtype)
      
    else:
      edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
                                           fill_value=opt['self_loop_weight'],
                                           num_nodes=data.num_nodes,
                                           dtype=data.x.dtype)
    self.odefunc.edge_index = edge_index.to(device)
    self.odefunc.edge_weight = edge_weight.to(device)
    self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

    if opt['adjoint']:
      from torchdiffeq import odeint_adjoint as odeint
    else:
      from torchdiffeq import odeint

    self.train_integrator = odeint
    self.test_integrator = odeint
    self.set_tol()


  def rewire(self, edge_index, x):
    if self.opt['data_norm'] == 'rw':
      
      self.odefunc.edge_index, self.odefunc.edge_weight = get_rw_adj(edge_index, edge_weight=None, norm_dim=1,
                                                                   fill_value=self.opt['self_loop_weight'],
                                                                   num_nodes=x.shape[0],
                                                                   dtype=x.dtype)
      
    else:
      self.odefunc.edge_index, self.odefunc.edge_weight = gcn_norm_fill_val(edge_index, edge_weight=None,
                                           fill_value=self.opt['self_loop_weight'],
                                           num_nodes=x.shape[0],
                                           dtype=x.dtype)

  def forward(self, x, new_edge_index):
    t = self.t.type_as(x)

    self.odefunc.edge_index = new_edge_index
    self.rewire(new_edge_index, x)


    integrator = self.train_integrator if self.training else self.test_integrator
    
    reg_states = tuple( torch.zeros(x.size(0)).to(x) for i in range(self.nreg) )

    func = self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc
    
    from run_GNN import Print
    # Print(func(x))
    state = (x,) + reg_states if self.training and self.nreg > 0 else x
    # func_with_edge_index = lambda 
    
    if self.opt["adjoint"] and self.training:
      
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
        adjoint_method=self.opt['adjoint_method'],
        adjoint_options=dict(step_size = self.opt['adjoint_step_size'], max_iters=self.opt['max_iters']),
        atol=self.atol,
        rtol=self.rtol,
        adjoint_atol=self.atol_adjoint,
        adjoint_rtol=self.rtol_adjoint)
    else:
      from run_GNN import Print
      # Print()
      state_dt = integrator(
        func, state, t,
        method=self.opt['method'],
        options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
        atol=self.atol,
        rtol=self.rtol)

    if self.training and self.nreg > 0:
      z = state_dt[0][1]
      reg_states = tuple( st[1] for st in state_dt[1:] )
      return z, reg_states
    else: 
      z = state_dt[1]
      return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
