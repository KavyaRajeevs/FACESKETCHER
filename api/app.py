
from flask import Flask, request, jsonify, send_file
from text_to_sketch import TextToSketchGenerator
import os
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)

# Configure CORS
CORS(app,
     resources={r"/*": {
         "origins": ["http://localhost:3000"],
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "supports_credentials": True,
         "expose_headers": ["Content-Type"],
         "allow_credentials": True,
         "max_age": 120
     }})

# Paths to trained model files (relative to this `api` directory)
TEXT_ENCODER_PATH = r'./ponnu_models/text_encoder600.pth'
GENERATOR_PATH = r'./ponnu_models/netG_epoch_600.pth'
CFG_PATH = r'./config/train_sketch_18_4.yml'

# Initialize the generator
generator = TextToSketchGenerator(TEXT_ENCODER_PATH, GENERATOR_PATH, CFG_PATH)


# Home route
@app.route('/')
def home():
    return jsonify({"message": "Sketching API is running!"})


# API route to generate a sketch
@app.route('/generate_sketch', methods=['POST', 'OPTIONS'])
def generate_sketch():

    if request.method == 'OPTIONS':
        return jsonify({"message": "Preflight OK"}), 200

    try:
        data = request.json

        if not data or 'description' not in data:
            return jsonify({"error": "Missing 'description' field"}), 400

        text_description = data['description']
        print(f"Received description: {text_description}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        output_dir = r"./api/generated_sketches"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
            f"generated_sketch_{timestamp}.png"
        )

        print("Generating sketch...")
        generator.generate_sketch(text_description, save_path=output_path)
        print("Sketch generated successfully")

        return send_file(
            output_path,
            mimetype='image/png',
            as_attachment=False,
            download_name=f"generated_sketch_{timestamp}.png"
        )

    except Exception as e:
        print("Error generating sketch:", str(e))

        return jsonify({
            "error": str(e),
            "message": "Failed to generate sketch"
        }), 500


# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })


# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        "error": "Not Found",
        "message": "The requested resource was not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred"
    }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))

    app.run(
        host='127.0.0.1',
        port=port,
        debug=True
    )