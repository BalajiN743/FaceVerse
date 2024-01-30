from faceverse_cuda.FaceVerseModel import FaceVerseModel
import numpy as np

def get_faceverse(**kargs):
    
    faceverse_dict = np.load(model_path, allow_pickle=True).item()
    #for i in range(52):
    #    print(i, faceverse_dict['exp_name_list_52'][i])
    faceverse_model = FaceVerseModel(faceverse_dict, **kargs)
    return faceverse_model, faceverse_dict

