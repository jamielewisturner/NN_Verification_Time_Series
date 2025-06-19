### Copy of code added to alpha-beta-CROWN/complete_verifier/specifications.py


class CustomSpec(Specification):
    def construct_vnnlib(self, dataset, x_range, example_idx_list):
        vnnlib = []

        for key, value in dataset.items():
            try:
                print(f"    {key}: type = {type(value)}, shape =", getattr(value, "shape", None))
            except Exception as e:
                print(f"    {key}: could not print shape ({e})")


        delta = arguments.Config['specification']['delta']

        print("Using delta=",delta)
        vnnlib = []

        # Assume dataset['orig_output'] is a Tensor of shape (N, num_outputs),
        # precomputed by running the model on X before any perturbation.
        clean_and_y = dataset['labels']  # shape: (N, num_outputs)
        orig_all = clean_and_y[:, :4] # shape: (N, num_outputs)
        num_out = self.num_outputs        # e.g. 4

        for k, i in enumerate(example_idx_list):
            this_x_range = x_range[k]      # shape (2, flattened_input_dim)
            orig_vec = orig_all[i]         # shape: (num_outputs,)
            new_c = []

            # For each output index j, enforce two inequalities:
            for j in range(num_out):
                # (1)  w_j >= orig_vec[j] - delta
                C_pos = torch.zeros(1, num_out)
                C_pos[0, j] = 1.0
                rhs_pos = np.array([orig_vec[j].item() - delta], dtype=np.float32)
                new_c.append((C_pos, rhs_pos))

                # (2)  w_j <= orig_vec[j] + delta  <=>  -w_j >= -orig_vec[j] - delta
                C_neg = torch.zeros(1, num_out)
                C_neg[0, j] = -1.0
                rhs_neg = np.array([-orig_vec[j].item() - delta], dtype=np.float32)
                new_c.append((C_neg, rhs_neg))


            vnnlib.append([(this_x_range, new_c)])

        return vnnlib
    

class MRSpec(Specification):
    def construct_vnnlib(self, dataset, x_range, example_idx_list):
        vnnlib = []

        for key, value in dataset.items():
            try:
                print(f"    {key}: type = {type(value)}, shape =", getattr(value, "shape", None))
            except Exception as e:
                print(f"    {key}: could not print shape ({e})")


        delta = arguments.Config['specification']['delta']

        print("Using delta=",delta)
        vnnlib = []

        # Assume dataset['orig_output'] is a Tensor of shape (N, num_outputs),
        # precomputed by running the model on X before any perturbation.
        clean_and_y = dataset['labels']  # shape: (N, num_outputs)

        orig_all = clean_and_y[:, :4]               # shape [10, 4]
        #   – slice off the remaining 20 and reshape → b
        y_all = clean_and_y[:, 4:]          # shape [10, 20]
        y_all = y_all.view(y_all.shape[0], 5, 4)  

        for k, i in enumerate(example_idx_list):
            this_x_range = x_range[k]      # shape (2, flattened_input_dim)
            orig_vec = orig_all[i]         # shape: (num_outputs,)
            returns_mat = y_all[i] 
            new_c = []

            mean_ret = returns_mat.mean(dim=0)

            orig_avg = (orig_vec * mean_ret).sum().item()

            C = mean_ret.unsqueeze(0) 

            rhs = np.array([orig_avg - delta], dtype=np.float32)

            new_c.append((C, rhs))


            vnnlib.append([(this_x_range, new_c)])

        return vnnlib
