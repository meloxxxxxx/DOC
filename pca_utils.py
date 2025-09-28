import torch
import numpy as np

class pcaMonitor:
    def __init__(self, k_start=48, k_add=48, drift_eps=0.05, add_eps=0.1, l=2, max_drift_bias = 0.01, p = 1):
        """
        Args:
            k_start: 初始主成分数量
            k_add: 每次新任务增加的成分名额
            eps: 残差阈值比例
        """
        self.k_start = k_start
        self.k = k_start  # maximum principal component number
        self.k_add = k_add
        self.drift_eps = drift_eps
        self.add_eps = add_eps
        self.l = l  # amnestic factor
        self.p = p  
        self.max_drift_bias = max_drift_bias
        
        self.task_id = 0 
        self.total_data_n = 0  
        self.task_data_n = 0
        
        self.v: list[torch.Tensor] = []  # principal component list
        self.component_n = {}
        self.task_rates: list[rate_info] = [rate_info(self)]
        
        self.h_norm2_avg = 0 
        self.residual_norm = 1
        self.h_norm = 1
        
        self.historical_component_ids = []
        self.h_shape = None

    def new_task(self, k_add=None, drift_eps=None, add_eps=None, l=None, drift_bias = None):
        self.task_id += 1
        self.task_data_n = 0

        self.k_add = k_add if k_add else self.k_add
        self.drift_eps = drift_eps if drift_eps else self.drift_eps
        self.add_eps = add_eps if add_eps else self.add_eps
        self.max_drift_bias = drift_bias if drift_bias else self.max_drift_bias
        self.l = l if l else self.l
        self.k = len(self.v) + self.k_add 
        
        self.task_rates.append(rate_info(self))
        self.historical_component_ids = range(len(self.v))

    def compute_delta_hs(self, params: dict, grads: dict, hidden_states: dict) -> torch.Tensor:
        delta_hs = {}
        for layer_name in hidden_states:
            hidden_state_avg = hidden_states[layer_name].mean(dim=(0, 1)).unsqueeze(0).T
            for name in params:
                if name.startswith(layer_name + '.') and ('.lora_B' in name):
                    #print(f'computing delta h of {name}')
                    b_name = name
                    a_name = name.replace('.lora_B', '.lora_A')
                    B = params[b_name]
                    A = params[a_name]
                    B_grad = grads[b_name]
                    A_grad = grads[a_name]
                    delta_h = B_grad @ (A @ hidden_state_avg) + B @ (A_grad @ hidden_state_avg)
                    if '.q_proj' in name:
                        delta_hs[layer_name + '.self_attn.q_proj'] = delta_h
                    elif '.v_proj' in name:
                        delta_hs[layer_name + '.self_attn.v_proj'] = delta_h

        if self.h_shape == None:
            delta_hs_vector, self.h_shape = flatten_parameters(delta_hs, return_shape=True)
        else:
            delta_hs_vector = flatten_parameters(delta_hs)
        
        return delta_hs_vector

    def fit(self, delta_hs_vector: torch.Tensor):
        self.h_norm = delta_hs_vector.norm(p='fro').item()
        self.total_data_n += 1
        self.task_data_n += 1
        
        eta = min(1+self.l, max(self.total_data_n-1, 1)) / (self.total_data_n)
        self.h_norm2_avg = (1-eta) * self.h_norm2_avg + eta * (self.h_norm**2)

        scalar = np.sqrt(self.h_norm2_avg)
        delta_hs_vector = (1 / (scalar+1e-8)) * delta_hs_vector
        residual_h = delta_hs_vector.clone()
        self.h_norm = self.h_norm / (scalar + 1e-8)
        self.residual_norm = self.h_norm

        for i in range(len(self.v)):
            n_i = self.component_n[i]
            eta = min(1+self.l, max(n_i-1, 1)) / (n_i)

            dot = torch.sum(residual_h * self.v[i-1]).item()
            norm = self.v[i-1].norm(p='fro').item()
            
            # v_i = (1-eta) * v_i + (dot / norm_v_i) * eta * h
            new_v_i = (1-eta) * self.v[i-1] +  (eta * dot / (norm + 1e-8)) * residual_h
            self.v[i-1] = new_v_i

            dot = torch.sum(residual_h * self.v[i-1]).item()
            norm = self.v[i-1].norm(p='fro').item()

            # h = h - dot / (norm**2 + 1e-8) * v_i
            residual_h = residual_h - (dot / (norm**2 + 1e-8)) * self.v[i-1]

            self.component_n[i] += 1

        self.residual_norm = residual_h.norm(p='fro').item()
        
        if ((self.drift_eps <= self.get_residual_rate() < self.add_eps)) or (self.get_residual_rate() >= self.add_eps and len(self.v) >= self.k):
            #加入drift
            drift_bias = self.max_drift_bias * ((self.get_residual_rate()) ** self.p)
            self.drift(delta_hs_vector, drift_bias)

        elif (self.get_residual_rate() >= self.add_eps and len(self.v) < self.k):
            self.v.append((1 / (self.residual_norm+1e-8)) * residual_h)
            self.component_n[len(self.v)-1] = 1
            
            print(f"Added new component.\ncomponents remaining: {self.k-len(self.v)}")
        
        self.update_rates(delta_hs_vector)

    def drift(self, normalized_delta_hs_vector: torch.Tensor, drift_bias):
        print(f'residual rate {self.get_residual_rate()}, drift_bias {drift_bias}, drifting components...')
        residual_h = normalized_delta_hs_vector.clone()
        for i in range(len(self.v)):
            dot = torch.sum(residual_h * self.v[i-1]).item()
            norm = self.v[i-1].norm(p='fro').item()
            
            # v_i = (1-drift_bias) * v_i + (dot / norm_v_i) * drift_bias * h
            new_v_i = (1-drift_bias) * self.v[i-1] + (drift_bias * dot / (norm + 1e-8)) * residual_h
            self.v[i-1] = new_v_i

            dot = torch.sum(residual_h * self.v[i-1]).item()
            norm = self.v[i-1].norm(p='fro').item()

            # h = h - dot / (norm**2 + 1e-8) * v_i
            residual_h = residual_h - (dot / (norm**2 + 1e-8)) * self.v[i-1]

        self.residual_norm = residual_h.norm(p='fro').item()

    def update_rates(self, h: torch.Tensor):
        h_norm = h.norm(p='fro').item()
        for i, v_i in enumerate(self.v):
            v_i_norm = v_i.norm(p='fro').item()
            component_cos = torch.sum(h * v_i).item() / (v_i_norm * h_norm +1e-8)
            self.task_rates[self.task_id].update_rates(i, component_cos)

    def cut_one_component(self, grads: dict, component_id: int) -> dict:
        component_to_cut = unflatten_parameters(self.v[component_id], self.h_shape)
        
        for module_name in component_to_cut:
            V = component_to_cut[module_name]
            for name in grads:
                if name.startswith(module_name + '.') and ('.lora_B' in name):
                    #print(f'cutting {module_name}')
                    b_name = name
                    B_grad = grads[b_name]
                    
                    projection_coeff = (V.T @ B_grad) / (V.norm(p='fro').item() + 1e-8)

                    # 投影矩阵：v * projection_coeff，形状 [m, n]
                    projection = V @ projection_coeff
                    
                    # 减去投影
                    grads[b_name] = B_grad - projection
                
                    break

        return grads

    def cut_component_helper(self, delta_hs_vector: dict) -> list:
        freeze_component_ids = [i for i in self.historical_component_ids]
        task_rate_matrix = self.get_rate_info()
        all_positive = (task_rate_matrix > 0).all(dim=0)
        all_negative = (task_rate_matrix < 0).all(dim=0)
        for i, positive in enumerate(all_positive):
            if torch.sum(delta_hs_vector * self.v[i]).item()>0 and positive:
                freeze_component_ids.remove(i)
        for i, negative in enumerate(all_negative):
            if torch.sum(delta_hs_vector * self.v[i]).item()<0 and negative:
                freeze_component_ids.remove(i)    
        return freeze_component_ids

    def cut_components(self, grads: dict, delta_hs_vector: dict):
        freeze_component_ids = self.cut_component_helper(delta_hs_vector)
        for i in freeze_component_ids:
            grads = self.cut_one_component(grads, i)
        return grads

    def get_rate_info(self) -> torch.Tensor:
        task_rate_matrix = torch.zeros([self.task_id+1, len(self.task_rates[self.task_id].rates)])
        for task_id, task_component in enumerate(self.task_rates):
            for i, rate in task_component.get_rates().items():
                task_rate_matrix[task_id, i] = rate
        return task_rate_matrix

    def get_components(self) -> list[torch.Tensor]:
        return self.v
    
    def get_residual_rate(self):
        return self.residual_norm / (self.h_norm + 1e-8)

    def get_status(self):
        return {
                'task_id': self.task_id,
                'num_components': len(self.v), 
                'component_capasity': self.k
                }

class rate_info:
    def __init__(self, pca_monitor: pcaMonitor):
        self.rates = {}
        self.history_n = {}
        self.pca_monitor = pca_monitor
    def update_rates(self, component_n, new_rate):
        if component_n not in self.rates:
            self.rates[component_n] = new_rate
            self.history_n[component_n] = 1
        else:
            n = self.history_n[component_n]
            eta = min(1+self.pca_monitor.l, max(n-1, 1)) / (n)
            rate = self.rates[component_n]
            self.rates[component_n] = (1-eta) * rate + eta * new_rate
            self.history_n[component_n]+=1
    def get_rates(self):
        return self.rates

def flatten_parameters(params_dict, return_shape = False):
    shapes = {name: param.shape for name, param in params_dict.items()}
    flat_vector = torch.cat([param.flatten() for param in params_dict.values()])
    if return_shape:
        return flat_vector, shapes
    else:
        return flat_vector

def unflatten_parameters(flat_vector, shapes):
    params_dict = {}
    start = 0
    for name, shape in shapes.items():
        end = start + torch.prod(torch.tensor(shape)).item()
        params_dict[name] = flat_vector[start:end].reshape(shape)
        start = end
    return params_dict

def register_hidden_state_hook(model: torch.nn.Module):
    hidden_state_container = {}
    layers = model.base_model.model.model.layers #for LLaMA only
    
    def hidden_state_hook_fn(module, input, output):
        with torch.inference_mode():
            layer_name = None
            for name, m in model.named_modules():
                if m is module:
                    layer_name = name
                    break
            
            hidden_state = input[0] if isinstance(input, tuple) else input
            hidden_state_container[layer_name] = hidden_state
    
    hidden_state_hooks = [layer.register_forward_hook(hidden_state_hook_fn) for layer in layers]
    return hidden_state_container, hidden_state_hooks

def get_parameters(model: torch.nn.Module):
    params = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            params[name] = param.clone()
    return params

def get_grads(model: torch.nn.Module):
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone()
    return grads

def replace_grads(model: torch.nn.Module, new_grads: dict):
    for name, param in model.named_parameters():
        if name in new_grads:
            param.grad = new_grads[name]

