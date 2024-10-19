# Speech-to-Text AI Assistant

A Python application that converts speech to text using OpenAI's Whisper model, stores conversations in a FAISS vector database, and generates responses using GPT-4.

## Features

- Speech-to-text conversion using Whisper
- Vector similarity search using FAISS
- Response generation using GPT-4
- Web interface using Gradio

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/speech-to-text-assistant.git
cd speech-to-text-assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Activate the virtual environment:
```bash
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Run the application:
```bash
python src/assistant.py
```

3. Open your browser and go to the URL shown in the terminal (usually http://localhost:7860)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)