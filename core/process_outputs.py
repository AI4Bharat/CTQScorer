import time
from tqdm.auto import tqdm
from utils.constants import BATCH_SIZE

def post_process_output(output):
    output = output.split('\n###')[0]
    output = output.split('###')[0]
    output = output.strip().strip('"').replace('\n', '')
    return output

"""
Given the model and an object with prompts, this function predicts the outputs in a batch
and writes it to the file and returns the list of outputs.
"""
def predict_outputs(pipe, datasetObj, prediction_file, model_name=''):
    start_time = time.time()
    pred_dst = []
    for start_index in tqdm(range(0, len(datasetObj), BATCH_SIZE)):
        
        for output in pipe(datasetObj.prompts[start_index: start_index + BATCH_SIZE], 
        max_new_tokens=datasetObj.getNumTokens(start_index), batch_size=BATCH_SIZE):
            
            # clean up the output obtained from model for evaluation purpose
            output = post_process_output(output[0].get('generated_text'))
            
            # print(output)
            pred_dst.append(output)

            # capture output obtained from model
            with open(prediction_file, 'a') as f:
                f.write('{}\n'.format(output))
        
    end_time = time.time()
    print('Time for one set: {}'.format(round(end_time - start_time, 3)))
    return pred_dst