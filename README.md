**TRAINING** 
Run_Training.py: Main file to start the training process. It needs:
                1. log name
                2. dataset_dir -> directory containing clean and scan images
                3. valid_path -> directory containing clean and scan images for validation
                4. color_encoder_path -> trained model file for color encoder
                5. diffusion_weights_path [OPTIONAL]-> trained diffusion model for retraining

**TEST**
Sampling_descan.py: Main file to test the model. It need: 
                1. diffusion_weights_path -> path of trained diffusion model
                2. color_encoder_path -> path of trained color correction model
                3. test_folder -> directory containing all the test images
                
