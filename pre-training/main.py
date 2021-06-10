# load general packages and functions
import datetime

# load program-specific functions
import util

util.suppress_warnings()
from parameters.constants import constants as C
from Workflow import Workflow

# defines and runs the job



def main():
    """ Defines the type of job (preprocessing, training, generation, testing, multiple validation, or computation of the validation loss), 
    runs it, and writes the job parameters used.
    """
    # fix date/time
    _ = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    workflow = Workflow(constants=C)

    job_type = C.job_type
    print(f"* Run mode: '{job_type}'", flush=True)

    if job_type == "preprocess":
        # write preprocessing parameters
        util.write_preprocessing_parameters(params=C)

        # preprocess all datasets
        workflow.preprocess_phase()

    elif job_type == "train":
        # write training parameters
        util.write_job_parameters(params=C)

        # train model and generate graphs
        workflow.training_phase()

    elif job_type == "generate":
        # write generation parameters
        util.write_job_parameters(params=C)

        # generate molecules only
        workflow.generation_phase()

    elif job_type == "test":
        # write testing parameters
        util.write_job_parameters(params=C)

        # evaluate best model using the test set data
        workflow.testing_phase()
    
    elif job_type == "multiple_valid":
        # write multiple validation parameters
        util.write_job_parameters(params=C)

        # evaluate best model using the valid set data
        workflow.multiple_valid_phase()

    elif job_type == "valid_loss":
        # write valid loss parameters
        util.write_job_parameters(params=C)

        # generate molecules only
        workflow.compute_valid_loss()

    else:
        return NotImplementedError("Not a valid `job_type`.")


if __name__ == "__main__":
    main()
