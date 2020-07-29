annotator.py: genera i file di annotazione per training set e validation set da utilizzare per la Plate Detection

extract_plates.py: estrae le targhe dalle immagini e le raddrizza per costruire il dataset "cropped" per la fase di character recognition

gt_split.py: separa i file della gt relativi a training set e validation set in accordo alla suddivisione fatta dall'annotator

inspect_plates_data.ipynb: verifica del dataset costruito

inspect_plates_model.ipynb: verifica dei risultati dell'addestramento

plate_detection.py: detection delle targhe sul dataset in input, restituisce le immagini annotate e i punti estremi della targa

plates.py: definizione dei dataset, tecniche di augmentation e procedura di training

training.ipynb: training della rete