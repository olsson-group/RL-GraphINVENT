# load general packages and functions
import numpy as np
import pickle
import shutil
import time
import torch
import torch.utils.tensorboard
from tqdm import tqdm
import os
from rdkit.Chem import MolFromSmiles, QED, Crippen, Descriptors, rdMolDescriptors, AllChem
from rdkit import DataStructs
import copy
from tensorboard_writer import writer as tb_writer

# load program-specific functions
import analyze as anal
import generate
import util
from loss import compute_loss
from score import compute_score

# defines `Workflow` class



# set default torch dtype
torch.set_default_dtype(torch.float32)

class Workflow:
    """ Single `Workflow` class split up into different functions for
      1) preprocessing various molecular datasets
      2) training generative models
      3) generating molecules using pre-trained models
      4) evaluating generative models

    The preprocessing step reads a set of molecules and generates training data
    for each molecule in HDF file format, consisting of decoding routes and
    APDs. During training, the decoding routes and APDs are used to train graph
    neural network models to generate new APDs, from which actions are
    stochastically sampled and used to build new molecular graphs. During
    generation, a pre-trained model is used to generate a fixed number of
    structures. During evaluation, metrics are calculated for the test set.
    """
    def __init__(self, constants):

        self.start_time = time.time()

        self.C = constants

        # create placeholders
        self.agent_model = None
        self.prior_model = None
        self.optimizer = None

        self.current_epoch = None
        self.restart_epoch = None
        self.ts_properties = None

        self.tensorboard_writer = None

        self.n_subgraphs = None ##

        self.best_avg_score = 0

    def get_ts_properties(self):
        """ Loads the training sets properties from CSV as a dictionary, properties
        are used later for model evaluation.
        """
        filename = self.C.training_set[:-3] + "csv"
        self.ts_properties = util.load_ts_properties(csv_path=filename)

    def define_model_and_optimizer(self):
        """ Defines the model (`self.model`) and the optimizer (`self.optimizer`).
        """
        print("* Defining model and optimizer.", flush=True)
        job_dir = self.C.job_dir
        model_dir = self.C.dataset_dir

        if self.C.restart:
            print("-- Loading model from previous saved state.", flush=True)
            self.restart_epoch = util.get_restart_epoch()
            self.agent_model = torch.load(f"{job_dir}model_restart_{self.restart_epoch}.pth")
            self.prior_model = torch.load(f"{model_dir}pretrained_model.pth")
            self.prev_model = torch.load(f"{model_dir}pretrained_model.pth")
            # Load sklearn activity model
            with open(self.C.data_path + "qsar_model.pickle", 'rb') as file:
                model_dict = pickle.load(file)                                      
                self.drd2_model = model_dict["classifier_sv"]
            

            print(
                f"-- Backing up as "
                f"{job_dir}model_restart_{self.restart_epoch}_restarted.pth.",
                flush=True,
            )
            shutil.copyfile(
                f"{job_dir}model_restart_{self.restart_epoch}.pth",
                f"{job_dir}model_restart_{self.restart_epoch}_restarted.pth",
            )

        else:
            print("-- Initializing model from scratch.", flush=True)
            self.agent_model = torch.load(f"{model_dir}pretrained_model.pth")
            self.prior_model = torch.load(f"{model_dir}pretrained_model.pth")
            self.prev_model = torch.load(f"{model_dir}pretrained_model.pth")
            with open(self.C.data_path + "qsar_model.pickle", 'rb') as file:
                model_dict = pickle.load(file)                                      
                self.drd2_model = model_dict["classifier_sv"]

            self.restart_epoch = 0

        start_epoch = self.restart_epoch + 1
        end_epoch = start_epoch + self.C.epochs

        print("-- Defining optimizer.", flush=True)
        self.optimizer = torch.optim.Adam(
            params=self.agent_model.parameters(),
            lr=self.C.init_lr,
            weight_decay=self.C.weight_decay,
        )

        print("-- Defining scheduler.", flush=True)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr= self.C.max_rel_lr * self.C.init_lr,
            div_factor= 1. / self.C.max_rel_lr,
            final_div_factor = 1. / self.C.min_rel_lr,
            pct_start = 0.05,
            total_steps=self.C.epochs,
            epochs=self.C.epochs
        )

        return start_epoch, end_epoch


    def learning_phase(self):
        """ Trains model (`self.model`) and generates graphs.
        """
        self.get_ts_properties()

        self.initialize_output_files()

        start_epoch, end_epoch = self.define_model_and_optimizer()

        # Evaluate model before training
        score = self.evaluate_model()
        tb_writer.add_scalar("Evaluation/score", score, 0)
        with open(self.C.job_dir + "eval-scores.csv", "w") as output_file:
            output_file.write(f"Epoch \t Score\n")
            output_file.write(f"0 \t {score:.8f}\n")

        print("* Beginning learning.", flush=True)

        for epoch in range(start_epoch, end_epoch):

            self.current_epoch = epoch
            self.learning_step()

            # evaluate model every `sample_every` epochs (not every epoch)
            if epoch % self.C.sample_every == 0:
                score = self.evaluate_model()
                with open(self.C.job_dir + "eval-scores.csv", "a") as output_file:
                    output_file.write(f"{self.current_epoch} \t {score:.8f}\n")
                tb_writer.add_scalar("Evaluation/score", score, epoch)
                if score > self.best_avg_score:
                    self.best_avg_score = score
                    self.prev_model = copy.deepcopy(self.agent_model)
                    print("Updated best model", flush=True)

        self.print_time_elapsed()

    def generation_phase(self):
        """ Generates molecules from a pre-trained model.
        """
        self.get_ts_properties()

        model_dir = self.C.dataset_dir
        self.prior_model = torch.load(f"{model_dir}pretrained_model.pth")

        self.restart_epoch = self.C.generation_epoch
        print(f"* Loading model from previous saved state (Epoch {self.restart_epoch}).", flush=True)
        model_path = self.C.job_dir + f"model_restart_{self.restart_epoch}.pth"
        self.agent_model = torch.load(model_path)

        with open(self.C.data_path + "qsar_model.pickle", 'rb') as file:
            model_dict = pickle.load(file)                                      
            self.drd2_model = model_dict["classifier_sv"]

        self.agent_model.eval()
        with torch.no_grad():
            score = self.generate_graphs(n_samples=self.C.n_samples)

        self.print_time_elapsed()

    def evaluate_model(self):
        """ Evaluates model by calculating the UC-JSD from generated structures.
        Saves model scores in `validation.csv` and then saves model state.
        """
        self.agent_model.eval()      # sets layers to eval mode (e.g. norm, dropout)
        with torch.no_grad():  # deactivates autograd engine

            # generate graphs required for model evaluation
            # note that evaluation of the generated graphs happens in
            # `generate_graphs()`, and molecules are saved as `self` attributes
            
            print("* Evaluating model.", flush=True)
            score = self.generate_graphs(n_samples=self.C.n_samples, evaluation=True)

            # `pickle.HIGHEST_PROTOCOL` good for large objects
            if self.current_epoch != None:
                print(f"* Saving model state at Epoch {self.current_epoch}.", flush=True)
                model_path_and_filename = (self.C.job_dir + f"model_restart_{self.current_epoch}.pth")
                torch.save(obj=self.agent_model,
                        f=model_path_and_filename,
                        pickle_protocol=pickle.HIGHEST_PROTOCOL)

        return score

    def initialize_output_files(self):
        """ Creates output files (with appropriate headers) for new (i.e.
        non-restart) jobs. If restart a job, and all new output will be appended
        to existing output files.
        """
        if not self.C.restart:
            print("* Touching output files.", flush=True)
            # begin writing `generation.csv` file
            csv_path_and_filename = self.C.job_dir + "generation.csv"
            util.properties_to_csv(
                prop_dict=self.ts_properties,
                csv_filename=csv_path_and_filename,
                epoch_key="Training set",
                append=False,
            )

            # begin writing `convergence.csv` file
            util.write_model_status(append=False)

            # create `generation/` subdirectory to write generation output to
            os.makedirs(self.C.job_dir + "generation/", exist_ok=True)

    def generate_graphs(self, n_samples, evaluation=False):
        """ Generates `n_graphs` molecular graphs and evaluates them. Generates
        the graphs in batches the size of `self.C.batch_size` or `n_samples` (int),
        whichever is smaller.
        """
        print(f"* Generating {n_samples} molecules.", flush=True)

        generation_batch_size = min(self.C.gen_batch_size, n_samples)

        n_generation_batches = int(n_samples/self.C.gen_batch_size)
        if n_samples % self.C.gen_batch_size != 0:
            n_generation_batches += 1

        # generate graphs in batches
        score = torch.tensor([], device="cuda")
        for idx in range(0, n_generation_batches):
            print("Batch", idx +1, "of", n_generation_batches)

            # generate one batch of graphs
            # g : generated graphs (list of `GenerationGraph`s)
            # a : agent LLs (torch.Tensor)
            # p : prior LLs (torch.Tensor)
            # t : termination status (torch.Tensor)
            g, a, p, t = generate.build_graphs(agent_model=self.agent_model,
                                               prior_model=self.prior_model,
                                               n_graphs_to_generate=generation_batch_size,
                                               batch_size=generation_batch_size,
                                               write_mols=True)

            # analyze properties of new graphs and save results
            validity_tensor, smiles = anal.evaluate_generated_graphs(generated_graphs=g[0],
                                                                    termination=t,
                                                                    agent_lls=a,
                                                                    prior_lls=p,
                                                                    start_time=self.start_time,
                                                                    ts_properties=self.ts_properties,
                                                                    generation_batch_idx=idx)

            uniqueness_tensor = util.get_unique_tensor(smiles)
            score_batch = compute_score(g[1], t, validity_tensor, uniqueness_tensor, smiles, self.drd2_model)
            score = torch.cat((score, score_batch))

        return torch.mean(score)

    def print_time_elapsed(self):
        """ Prints elapsed time since input `start_time`.
        """
        stop_time = time.time()
        elapsed_time = stop_time - self.start_time
        print(f"-- time elapsed: {elapsed_time:.5f} s", flush=True)

    def learning_step(self):
        """ Performs one training epoch.
        """
        print(f"* Learning step {self.current_epoch}.", flush=True)
        self.agent_model.train()  # ensure model is in train mode

        # clear the gradients of all optimized `(torch.Tensor)`s
        self.agent_model.zero_grad()
        self.optimizer.zero_grad()
        

        # generate one batch of graphs
        # g : generated graphs (list of `GenerationGraph`s)
        # a : agent LLs (torch.Tensor)
        # p : prior LLs (torch.Tensor)
        # t : termination status (torch.Tensor)
        g, a, p, t = generate.build_graphs(agent_model=self.agent_model,
                                           prior_model=self.prior_model,
                                           n_graphs_to_generate=self.C.batch_size,
                                           batch_size=self.C.batch_size,
                                           mols_too=True
                                          )
    

        validity_tensor, smiles = anal.evaluate_generated_graphs(generated_graphs=g[0],
                                                                 termination=t,
                                                                 agent_lls=a,
                                                                 prior_lls=p,
                                                                 start_time=self.start_time,
                                                                 ts_properties=self.ts_properties,
                                                                 generation_batch_idx=-1)

        uniqueness_tensor = util.get_unique_tensor(smiles)
        score = compute_score(g[1], t, validity_tensor, uniqueness_tensor, smiles, self.drd2_model)
        loss = (1-self.C.alpha) * torch.mean(compute_loss(score, a, p, uniqueness_tensor))  

        score_write = torch.mean(torch.clone(score)).item()
        loss_write = torch.clone(loss)
        a_write = torch.clone(a)
        p_write = torch.clone(p)

        ## Generate graphs with prior model

        g, p, a, t = generate.build_graphs(agent_model=self.prev_model,
                                           prior_model=self.agent_model,
                                           n_graphs_to_generate=self.C.batch_size,
                                           batch_size=self.C.batch_size,
                                           mols_too=True,
                                           change_ap=True
                                          )
    

        validity_tensor, smiles = anal.evaluate_generated_graphs(generated_graphs=g[0],
                                                                 termination=t,
                                                                 agent_lls=a,
                                                                 prior_lls=p,
                                                                 start_time=self.start_time,
                                                                 ts_properties=self.ts_properties,
                                                                 generation_batch_idx=-1)

        uniqueness_tensor = util.get_unique_tensor(smiles)
        score = compute_score(g[1], t, validity_tensor, uniqueness_tensor, smiles, self.drd2_model)
        uniqueness_tensor = torch.where(score > self.best_avg_score, uniqueness_tensor, torch.zeros(len(score), device="cuda"))
        loss += self.C.alpha * torch.mean(compute_loss(score, a, p, uniqueness_tensor))  

        # backpropagate
        loss.backward()
        self.optimizer.step()

        # update the learning rate
        self.scheduler.step()


        util.write_model_status(
            epoch=self.current_epoch,
            lr=self.optimizer.param_groups[0]["lr"],
            loss=loss_write,
            score=score_write
        )
        util.tbwrite_nlls(
            epoch=self.current_epoch,
            agent_nlls=-a_write,
            prior_nlls=-p_write
        )