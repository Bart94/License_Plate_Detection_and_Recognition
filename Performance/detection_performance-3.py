import os, sys, cv2, argparse
from shapely.geometry import Polygon
from glob import glob
import numpy as np
import math

def load_gt(path):
    gt = {}
    for filename in glob(os.path.join(path, '*.txt')):
        with open(filename) as f:
            cont = 0
            imfile = os.path.basename(filename)
            print(imfile)
            gt[imfile] = []
            for line in f:
                    cont+=1
                    line = [x.strip() for x in line.split(',')]
                    line.pop(9)
                    line.pop(9)
                    points = []
                    for i in range(1, 5):
                        x = (float(line[i]))
                        y = (float(line[i+4]))
                        points.append({'x':x,'y':y})
                    points = {'points':points, 'score':None}
                    if imfile in gt:
                        gt[imfile].append(points)
                    else:
                        gt[imfile] = [points]
    return gt

def to_polygon(pts):
    return Polygon([(p['x'],p['y']) for p in pts]).buffer(0)

def init_performance(name, iou_th):
    #Inzializzazione delle performance (Nome, TP, FP, FN)
    return {'name': name, 'TP':0, 'FP':0, 'FN':0, 'IoU':0.0, 'IoU_samp':[], 'IoU_th':iou_th}

def compute_performance(performance, pred, true):
    #pred = [x['points'] for x in pred] 
    pred = [{'p':to_polygon(x['points']), 'matched':False} for x in pred]
    #true = [x['points'] for x in true]
    true = [{'p':to_polygon(x['points']), 'matched':False} for x in true]

    for p in pred:
        for t in true:
            if not t['matched'] and not p['matched']:
                try:
                    inter = p['p'].intersection(t['p']).area
                    union = p['p'].union(t['p']).area
                    iou = inter/union
                    

                    if iou > performance['IoU_th']:
                        performance['IoU'] = performance['IoU']+iou
                        performance['IoU_samp'].append(iou)
                        performance['TP'] += 1
                        t['matched'] = True
                        p['matched'] = True
                except:
                    for t in true:
                        if not t['matched']:
                            performance['FN'] += 1
                    print("Error computing performance")
                    
                   
        if not p['matched']:
            performance['FP'] +=1
    for t in true:
        if not t['matched']:
            performance['FN'] += 1
    #print_performance(performance)


def print_performance(performance):
    if performance['TP']+performance['FN'] == 0 or performance['TP']+performance['FP'] == 0:
        print('IoU %.2f'%(performance['IoU_th']))
        print('%s: no data' % performance['name'])
        return

    #Stampa le performance per ogni caso
    R = float(performance['TP'])/(performance['TP']+performance['FN'])
    P = float(performance['TP'])/(performance['TP']+performance['FP'])
    F = float(2*R*P/(R+P))
    if performance['TP']!= 0:
        IoU = performance['IoU']/performance['TP']
        IoU_diff = 0
        for iou in performance['IoU_samp']:
            IoU_diff += (iou-IoU)*(iou-IoU)
        IoU_std = math.sqrt(IoU_diff/performance['TP'])
        print('IoU %.2f'%(performance['IoU_th']))
        print('%s,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f'%(performance['name'], performance['TP'], performance['FP'], performance['FN'], R, P, F, IoU,IoU_std))
    else:
        print('IoU %.2f'%(performance['IoU_th']))
        print('%s,0,0,0,0,0,0'%(performance['name'] ))


def draw_points(image, plates, color, thickness):
    for plate in plates:
        p = plate['points']
        cv2.line(image, (int(p[0]['x']*image.shape[1]), int(p[0]['y']*image.shape[0])), (int(p[1]['x']*image.shape[1]),int(p[1]['y']*image.shape[0])), color, thickness)
        cv2.line(image, (int(p[1]['x']*image.shape[1]), int(p[1]['y']*image.shape[0])), (int(p[2]['x']*image.shape[1]),int(p[2]['y']*image.shape[0])), color, thickness)
        cv2.line(image, (int(p[3]['x']*image.shape[1]), int(p[3]['y']*image.shape[0])), (int(p[2]['x']*image.shape[1]),int(p[2]['y']*image.shape[0])), color, thickness)
        cv2.line(image, (int(p[3]['x']*image.shape[1]), int(p[3]['y']*image.shape[0])), (int(p[0]['x']*image.shape[1]),int(p[0]['y']*image.shape[0])), color, thickness)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detection Preformance')
    parser.add_argument("--dataset_path", type=str, nargs='?', default='platesmania6000', help="Path della cartella con le groundtruth.")
    parser.add_argument("--results_path", type=str, nargs='?', default='platesmania6000', help="Path della cartella con i risultati.")
    parser.add_argument("--id", type=str, nargs='?', default='prova', help="Id dell'esperimento.")
    parser.add_argument("--draw", type=int, nargs='?', default=0, help="Inserire un valore diverso da 0 per disegnare i risultati (GT in verde, risultati in rosso).")
    parser.add_argument("--save_result_path", type=str, nargs='?', default=None, help="Path della cartella in cui salvare le immagini annotate. Se vale None, non effettua il salvataggio.")
    parser.add_argument("--min", type=float, nargs='?', default=0.5, help="Valore minimo di IoU")
    parser.add_argument("--max", type=float, nargs='?', default=0.5, help="Valore massimo di IoU")
    parser.add_argument("--step", type=float, nargs='?', default=0.1, help="Step utilizzato per incrementare IoU")
    parser.add_argument("--result_file_path", type=str, nargs='?', default=None, help="Path del file su cui stampare i risultati")
    args = parser.parse_args()

    if(int(args.draw) != 0):
        cv2.namedWindow("detection", cv2.WINDOW_NORMAL)

    # Caricamento della gt
    gt = load_gt(args.dataset_path)

    # Caricamento dei risultati
    results = load_gt(args.results_path)

    # Inizializzazione della struttura performance
    performance = []
    start = float(args.min)
    end = float(args.max)
    step = float(args.step)
    while start <= end:
        performance.append(init_performance(args.id, start))
        start += step
    
    # Calcolo delle performance immagine per immagine
    for filename in glob(os.path.join(args.dataset_path, '*.txt')):
        imfile = os.path.basename(filename)
        
        # Fornire i risultati (secondo parametro) nello stesso formato di GT - Vedere funzione load_gt o esempio
        for p in performance:
            compute_performance(p, results[imfile], gt[imfile])
        print("Processed "+imfile)

        if(int(args.draw) != 0 or args.save_result_path != None):
            im_path = filename[:-4]+'.jpg'
            print(im_path)
            image = cv2.imread(im_path)
            draw_points(image, gt[imfile], (0,255,0), 6)
            draw_points(image, results[imfile], (0,0,255), 3)
            if(args.save_result_path != None):
                cv2.imwrite(os.path.join(args.save_result_path+os.path.basename(filename)[:-4]+'_output.jpg'), image.astype(np.uint8))
            if(int(args.draw) != 0):
                cv2.imshow("detection", image)
                if cv2.waitKey(0) & 0xff == ord('q'):
                    break

    # Stampa dei risultati complessivi
    if(args.result_file_path != None):
        sys.stdout = open(args.result_file_path, "w")
    for p in performance:
        print_performance(p)
    sys.stdout = sys.__stdout__