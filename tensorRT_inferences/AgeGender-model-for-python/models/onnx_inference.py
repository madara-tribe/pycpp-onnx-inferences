import onnxruntime
import onnx

def age_onnx_inference(crop_faces, onnx_model_path):
    image = crop_faces
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    generation = session.get_outputs()[0].name
    iden = session.get_outputs()[1].name
    gender_output = session.get_outputs()[2].name
    #print('image shape', image.shape)
    pred_generation, pred_iden, pred_gender = session.run([generation, iden, gender_output], {input_name: image})
    return pred_generation, pred_iden, pred_gender


def gender_onnx_inference(crop_faces, onnx_model_path):
    image = crop_faces
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    #print('image shape', image.shape)
    pred_gender = session.run(None, {input_name: image})[0]
    return pred_gender
