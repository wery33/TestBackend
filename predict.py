from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import tensorflow as tf
from flask_cors import CORS
import base64
from flask import send_file

app = Flask(__name__)
CORS(app)

# Function to perform image fusion
def fuse_images(image1, image2):
    # Convert PIL Images to numpy arrays
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # Get the dimensions of the images
    height, width, _ = image1_array.shape

    # Split the images into left and right halves
    half_width = width // 2
    image1_left_half = image1_array[:, :half_width, :]
    image1_right_half = image1_array[:, half_width:, :]
    image2_left_half = image2_array[:, :half_width, :]
    image2_right_half = image2_array[:, half_width:, :]

    # Fuse the images
    fused_image_1_to_2 = np.concatenate([image1_left_half, image2_right_half], axis=1)
    fused_image_2_to_1 = np.concatenate([image2_left_half, image1_right_half], axis=1)

    # Convert numpy array back to PIL Image
    fused_image_1_to_2 = Image.fromarray(fused_image_1_to_2)
    fused_image_2_to_1 = Image.fromarray(fused_image_2_to_1)

    return {"success": True, "fused_image_1_to_2": fused_image_1_to_2, "fused_image_2_to_1": fused_image_2_to_1}

@app.route('/image-fusion', methods=['POST'])
def image_fusion():
    try:
        # Check if both images are present in the request
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({"success": False, "error": "Both images are required."})

        # Get the uploaded images
        image1 = request.files['image1']
        image2 = request.files['image2']

        # Convert the images to PIL Image objects
        image1 = Image.open(image1)
        image2 = Image.open(image2)

        # Perform image fusion
        result = fuse_images(image1, image2)

        # Save fused images to temporary files
        fused_image_1_to_2_path = "fused_image_1_to_2.jpg"
        fused_image_2_to_1_path = "fused_image_2_to_1.jpg"
        result["fused_image_1_to_2"].save(fused_image_1_to_2_path)
        result["fused_image_2_to_1"].save(fused_image_2_to_1_path)

        return jsonify({
            "success": True,
            "fused_image_1_to_2_path": fused_image_1_to_2_path,
            "fused_image_2_to_1_path": fused_image_2_to_1_path
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/get-fused-image/<path:image_name>')
def get_fused_image(image_name):
    try:
        # Return the fused image file
        return send_file(image_name, mimetype='image/jpeg')

    except Exception as e:
        return str(e)

# Function to convert PIL Image to base64-encoded string
def pil_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.route('/download-fused-image/<path:image_name>')
def download_fused_image(image_name):
    try:
        # Return the fused image file
        return send_file(image_name, as_attachment=True)

    except Exception as e:
        return str(e)
    



def get_model():
    global model 
    model = tf.keras.models.load_model("tomatonew100.h5")
    print(" *Model Loaded")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image =np.expand_dims(image, axis = 0)

    return image

print(" * Loading Keras model...")
get_model()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        message = request.get_json(force= True)
        encoded = message['image']
        decoded = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(decoded))
        processed_image = preprocess_image(image, target_size= (256,256))
        prediction = model.predict(processed_image)[0]

        class_list = [
            'Tomato_Bacterial_spot_Tomato_Early_blight',
            'Tomato_Early_blight_Tomato__Target_Spot',
            'Tomato_Late_blight_Tomato_Bacterial_spot',
            'Tomato_Late_blight_Tomato__Target_Spot',
            'Tomato_Late_blight_Tomato_healthy',
            'Tomato_Leaf_Mold_Tomato_Bacterial_spot',
            'Tomato_Leaf_Mold_Tomato_Early_blight',
            'Tomato_Leaf_Mold_Tomato_YellowLeaf__Curl_Virus',
            'Tomato_Leaf_Mold_Tomato_healthy',
            'Tomato_Septoria_leaf_spot_Tomato_Bacterial_spot',
            'Tomato_Septoria_leaf_spot_Tomato_Late_blight',
            'Tomato_Septoria_leaf_spot_Tomato_Leaf_Mold',
            'Tomato_Septoria_leaf_spot_Tomato_YellowLeaf__Curl_Virus',
            'Tomato_Septoria_leaf_spot_Tomato__Target_Spot',
            'Tomato_Septoria_leaf_spot_Tomato_healthy',
            'Tomato_YellowLeaf__Curl_Virus_Tomato_spotted_spider_mite',
            'Tomato_mosaic_virus_Tomato_Bacterial_spot',
            'Tomato_mosaic_virus_Tomato_Early_blight',
            'Tomato_spotted_spider_mite_Tomato__Target_Spot',
            'Tomato_spotted_spider_mite_Tomato_healthy',
            'None'
        ]

        mapped_predictions = []
        max_confidence_idx = np.argmax(prediction)
        max_confidence_class = class_list[max_confidence_idx]
        max_confidence_value = round(float(prediction[max_confidence_idx]) * 100, 2)

        for idx, confidence in enumerate(prediction):
            class_name = class_list[idx]
            confidence = round(float(confidence) * 100, 2)
            mapped_predictions.append({'class_name': class_name, 'confidence': confidence})
        response_data = {
            'predictions': mapped_predictions,
            'max_confidence': {'class_name': max_confidence_class, 'confidence': max_confidence_value}
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)