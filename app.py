#Imports and libraries to be utilised.
import os
import time
import glob
import csv
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor, as_completed

#For handling OOM errors (for smaller GPUs like L4, T4 etc)
import torch

app = Flask(__name__)

UPLOADFOLDER = "./uploads" #Creates a directory (locally)
DATASET_FOLDER = "./data"  #default sample dataset location (optional)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'} #Supported file types

app.config['UPLOAD_FOLDER'] = UPLOADFOLDER #Flask config

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#############################
# Dataset Handling Functions
#############################
def validate_dataset_format(dataset_path: str, modality: str) -> None:
    """
    Ensure dataset directory is structured as for classification tasks:
    data/
      class1/
         file1.(jpg/png/mp4/avi)
      class2/
         file2.(jpg/png/mp4/avi)
    """
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} is not a directory.")
    
    # Each subdirectory should represent a class
    class_dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, d))]
    if not class_dirs:
        raise ValueError("No class subdirectories found in the dataset.")
    
    for class_dir in class_dirs:
        files = glob.glob(os.path.join(class_dir, "*"))
        if not files:
            raise ValueError(f"No files found in class directory {class_dir}.")
        for file_path in files:
            ext = os.path.splitext(file_path)[1].lower()
            if modality == "image" and ext not in {'.png', '.jpg', '.jpeg'}:
                raise ValueError(f"File {file_path} is not a valid image.")
            if modality == "video" and ext not in {'.mp4', '.avi'}:
                raise ValueError(f"File {file_path} is not a valid video.")

def load_dataset(dataset_path: str, modality: str):
    """
    Load dataset and return a list of (file_path, ground_truth) tuples.
    Ground truth is inferred from the subdirectory (class) name.
    """
    data = []
    class_dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, d))]
    for class_dir in class_dirs:
        label = os.path.basename(class_dir)
        for file_path in glob.glob(os.path.join(class_dir, "*")):
            data.append((file_path, label))
    return data

#############################
# Inference Functions with Fallback
#############################

# Video Inference Functions
def infer_with_gemini_video(file_path, prompt):
    from google import genai
    client = genai.Client(api_key="AIzaSyDYS6-_mECvwSsAfFswDgaC6hQI6azNx-I")
    print("Uploading video file...")
    video_file = client.files.upload(file=file_path)
    print(f"Completed upload: {video_file.uri}")
    while video_file.state.name == "PROCESSING":
        print('.', end='')
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)
    if video_file.state.name == "FAILED":
        raise ValueError(f"Upload failed: {video_file.state.name}")
    print('Done processing video file.')
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite-001",
        contents=[video_file, prompt]
    )
    return response.text

def infer_with_qwen_video(file_path, prompt):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import torch

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        video_url = f"file://{os.path.abspath(file_path)}"
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_url, "max_pixels": 360 * 420, "fps": 1.0},
                {"type": "text", "text": prompt}
            ]
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=1.0,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print("Qwen inference failed due to OOM. Falling back to Gemini for video.")
            return infer_with_gemini_video(file_path, prompt)
        else:
            raise e

# Image Inference Functions
def infer_with_qwen_image(file_path, prompt):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    import torch
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        image_url = f"file://{os.path.abspath(file_path)}"
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": prompt},
            ]
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print("Qwen image inference failed due to OOM. Falling back to Gemini for image.")
            return infer_with_gemini_image(file_path, prompt)
        else:
            raise e

def infer_with_gemini_image(file_path, prompt):
    from google import genai
    import PIL.Image
    image = PIL.Image.open(file_path)
    client = genai.Client(api_key="AIzaSyDYS6-_mECvwSsAfFswDgaC6hQI6azNx-I")
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite-001",
        contents=[prompt, image]
    )
    return response.text

#############################
# Flask Endpoints and Routing
#############################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # If a "data_directory" field is provided, we run in dataset mode
    data_dir = request.form.get('data_directory', '').strip()
    
    # Single file mode if no data directory provided
    if data_dir == "":
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        modality = request.form.get('modality')  # 'video' or 'image'
        models_selected = request.form.getlist('model')  # list of models
        prompt = request.form.get('prompt', '')
        expected_output = request.form.get('expected_output', '')  # optional expected output for single file
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        results = {}
        try:
            for model_name in models_selected:
                if modality == "video":
                    if model_name == "Gemini Flash Lite":
                        results[model_name] = infer_with_gemini_video(file_path, prompt)
                    elif model_name == "Qwen 2.5 3B Instruct VL":
                        results[model_name] = infer_with_qwen_video(file_path, prompt)
                    else:
                        results[model_name] = "Selected video model not supported."
                elif modality == "image":
                    if model_name == "Gemini Flash Lite":
                        results[model_name] = infer_with_gemini_image(file_path, prompt)
                    elif model_name == "Qwen 2.5 3B Instruct VL":
                        results[model_name] = infer_with_qwen_image(file_path, prompt)
                    else:
                        results[model_name] = "Selected image model not supported."
                else:
                    results[model_name] = "Unsupported modality."
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
        response = {'result': results}
        if expected_output:
            #Here, you could compute metrics such as accuracy, f1, etc. using sklearn
            from sklearn.metrics import classification_report
            #For single input, this might be trivial; we assume one ground truth per file.
            # In a real scenario, you’d compare lists of predictions and true labels.
            report = classification_report([expected_output], [list(results.values())[0]], output_dict=True)
            response['metrics'] = report
        
        return jsonify(response)
    
    # Dataset mode:
    else:
        modality = request.form.get('modality')  # 'video' or 'image'
        models_selected = request.form.getlist('model')  # list of models
        prompt = request.form.get('prompt', '')
        
        try:
            # Validate dataset structure
            validate_dataset_format(data_dir, modality)
            dataset = load_dataset(data_dir, modality)  # List of (file_path, ground_truth)
        except Exception as e:
            return jsonify({'error': f"Dataset error: {str(e)}"}), 400
        
        predictions = {}  # { file_path: { model_name: prediction } }
        ground_truths = {}  # { file_path: ground_truth }
        test_files = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_task = {}
            for file_path, ground_truth in dataset:
                ground_truths[file_path] = ground_truth
                test_files.append(file_path)
                for model_name in models_selected:
                    future = executor.submit(run_inference, file_path, modality, model_name, prompt)
                    future_to_task[future] = (file_path, model_name)
            
            for future in as_completed(future_to_task):
                file_path, model_name = future_to_task[future]
                try:
                    pred = future.result()
                except Exception as e:
                    pred = f"Error: {str(e)}"
                if file_path not in predictions:
                    predictions[file_path] = {}
                predictions[file_path][model_name] = pred
        
        # Aggregate results per model for metric evaluation
        metrics_results = {}
        from sklearn.metrics import classification_report
        for model_name in models_selected:
            y_true = []
            y_pred = []
            for file_path in test_files:
                y_true.append(ground_truths.get(file_path, ""))
                # If a model did not produce a prediction, treat it as an error
                y_pred.append(predictions[file_path].get(model_name, ""))
            # Compute the classification report for each model
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics_results[model_name] = report
        
        # Save detailed CSV of predictions and metrics
        csv_file = "evaluation_results.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["File", "Ground Truth", "Model", "Prediction"])
            for file_path in test_files:
                for model_name in models_selected:
                    writer.writerow([os.path.basename(file_path),
                                     ground_truths.get(file_path, ""),
                                     model_name,
                                     predictions[file_path].get(model_name, "")])
        
        return jsonify({
            'predictions': predictions,
            'metrics': metrics_results,
            'csv_file': csv_file
        })

def run_inference(file_path: str, modality: str, model_name: str, prompt: str) -> str:
    """
    Wrapper to run inference with the selected model.
    Includes fallback: if Qwen fails due to OOM, falls back to Gemini.
    """
    if modality == "video":
        if model_name == "Gemini Flash Lite":
            return infer_with_gemini_video(file_path, prompt)
        elif model_name == "Qwen 2.5 3B Instruct VL":
            try:
                return infer_with_qwen_video(file_path, prompt)
            except Exception as e:
                if "out of memory" in str(e).lower():
                    print("Falling back to Gemini (video) due to OOM in Qwen.")
                    return infer_with_gemini_video(file_path, prompt)
                else:
                    raise e
    elif modality == "image":
        if model_name == "Gemini Flash Lite":
            return infer_with_gemini_image(file_path, prompt)
        elif model_name == "Qwen 2.5 3B Instruct VL":
            try:
                return infer_with_qwen_image(file_path, prompt)
            except Exception as e:
                if "out of memory" in str(e).lower():
                    print("Falling back to Gemini (image) due to OOM in Qwen.")
                    return infer_with_gemini_image(file_path, prompt)
                else:
                    raise e
    return "Unsupported model/modality"

if __name__ == '__main__':
    if not os.path.exists(UPLOADFOLDER):
        os.makedirs(UPLOADFOLDER)
    app.run(debug=True)
