character_box_annotation.py: genera i file di annotazione per training set e validation set da utilizzare per la character recognition, considerando per ogni carattere della targa il box corrispondente

character_segmentation.py: utility per la segmentazione dei caratteri utilizzata per la costruzione del dataset "train cropped" usato per la character recognition

characters.py: definizione dei dataset, tecniche di augmentation e procedura di training

characters_detection.py: detection delle targhe sul dataset in input, restituisce le immagini annotate e i punti estremi della targa

gt_split.py: separa i file della gt relativi a training set e validation set in accordo alla suddivisione fatta da 'character_box_annotation.py'

training.ipynb: training della rete