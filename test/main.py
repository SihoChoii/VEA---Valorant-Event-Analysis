import torch
from demucs import pretrained
from demucs.apply import apply_model
import torchaudio
import os

def separate_audio(input_path, output_dir, model_name='htdemucs', device='cuda'):
    # Load the pre-trained model
    model = pretrained.get_model(model_name)
    
    # Move model to the specified device (cuda or cpu)
    model.to(device)
    
    # Load the audio file
    wav, sr = torchaudio.load(input_path)
    
    # Apply the model to the audio file
    with torch.no_grad():
        sources = apply_model(model, wav[None].to(device), shifts=1, split=True, overlap=0.25)[0]
    
    # Move the output back to CPU for saving
    sources = sources.cpu()
    
    # Save the separated audio files
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the sources: typically ['drums', 'bass', 'other', 'vocals']
    source_names = model.sources
    
    # Find the index of vocals and sum the rest for the instrumental
    vocals_index = source_names.index('vocals')
    vocals = sources[vocals_index]
    instrumental = sum(sources[i] for i in range(len(sources)) if i != vocals_index)
    
    # Save the vocal part
    vocal_path = os.path.join(output_dir, 'vocals.wav')
    torchaudio.save(vocal_path, vocals, sample_rate=sr)
    print(f'Saved vocals to {vocal_path}')
    
    # Save the instrumental part
    instrumental_path = os.path.join(output_dir, 'instrumental.wav')
    torchaudio.save(instrumental_path, instrumental, sample_rate=sr)
    print(f'Saved instrumental to {instrumental_path}')

# Example usage
if __name__ == '__main__':
    input_path = r"C:\Users\bench\Downloads\abyss.mp3"
    output_dir = 'separated'
    separate_audio(input_path, output_dir)
