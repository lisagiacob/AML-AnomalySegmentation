# AML Anomaly Segmentation Project
Group work for the course of Advanced Machine Learning at @POLITO

The task is to build tiny anomaly segmentation models to segment anomalies that could be deployed in real-time. 
The goal is for the models to be able to fit in small devices, which represents a realistic memory constraint for an edge application using a smart camera with some small onboard processing capacity.

## Goals 
1. Get acquainted with the task of anomaly segmentation and understand
the anomaly segmentation dataset and its complexities.
2. Run initial experiments with a few baseline models for anomaly
segmentation.
3. Analyze the anomaly segmentation results by training a void classifier
on different semantic segmentation architectures.
1. Propose your extensions to reduce anomaly segmentation model size while maintaining anomaly segmentation performance. You are free to propose your ideas and choose what you want to focus on.

## Letteratura
1. [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](BiSeNet.pdf)
2. [BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation](BiSeNetV2.pdf)
3. [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](ICNet.pdf)
4. [Enet: A deep neural network architecture for real-time semantic segmentation](Enet.pdf)
5. [SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation](SegmentMeIfYouCan.pdf)
6. ERFNet: Eﬃcient Residual Factorized ConvNet for Real-Time Semantic Segmentation


# Spiegazione e Confronto di BiSeNet, BiSeNetV2, ICNet e ENet

## Introduzione
La semantic segmentation è un compito fondamentale nel campo della visione artificiale, utilizzato in applicazioni come guida autonoma, realtà aumentata e robotica. Le reti BiSeNet, BiSeNetV2, ICNet ed ENet sono state progettate per bilanciare accuratezza e velocità, rendendole adatte per applicazioni in tempo reale. Questo documento confronta le caratteristiche principali di queste reti.

---

## BiSeNet
**Bilateral Segmentation Network (2018)**

- **Architettura**:
  - Due percorsi principali:
    1. **Spatial Path**: Preserva dettagli spaziali ad alta risoluzione.
    2. **Context Path**: Estrae informazioni contestuali globali.
  - Combina i due percorsi tramite un modulo di fusione.
- **Vantaggi**:
  - Alta accuratezza bilanciata con velocità.
  - Adatta a molte applicazioni generiche.
- **Limiti**:
  - Richiede risorse hardware moderate per ottenere velocità real-time.
- **Prestazioni**:
  - Buon equilibrio tra accuratezza e velocità.
  - ~65 FPS su immagini di 1024x512 con una GPU moderna.

---

## BiSeNetV2
**Bilateral Segmentation Network V2 (2020)**

- **Architettura**:
  - Ottimizzazione dell'approccio BiSeNet con:
    - **Detailed Branch**: Per catturare dettagli spaziali.
    - **Semantic Branch**: Per informazioni contestuali globali.
    - **Aggregation Module**: Fusione efficiente delle informazioni.
  - Design più leggero e veloce rispetto a BiSeNet.
- **Vantaggi**:
  - Altissima velocità (fino a 156 FPS su GPU moderne).
  - Ideale per dispositivi con risorse computazionali limitate.
- **Limiti**:
  - Leggero compromesso in termini di accuratezza rispetto a BiSeNet.
- **Prestazioni**:
  - Estremamente veloce (~156 FPS su immagini di 1024x512 con GPU moderne).

---

## ICNet
**Image Cascade Network (2018)**

- **Architettura**:
  - Struttura a cascata con tre rami:
    1. **Low-resolution branch**: Per catturare il contesto globale.
    2. **Medium-resolution branch**: Raffina le informazioni.
    3. **High-resolution branch**: Preserva dettagli spaziali.
  - Utilizza la fusione delle caratteristiche tramite Cascade Feature Fusion (CFF).
- **Vantaggi**:
  - Eccellente velocità grazie all'approccio multi-risoluzione.
  - Adatta a dispositivi mobili o scenari con risorse limitate.
- **Limiti**:
  - Accuratezza moderata rispetto a reti più complesse.
- **Prestazioni**:
  - ~30 FPS su dispositivi mobili.
  - ~60 FPS su GPU moderne.

---

## ENet
**Efficient Neural Network (2016)**

- **Architettura**:
  - Rete altamente compressa, progettata specificamente per la velocità.
  - Utilizza convoluzioni più leggere e meno parametri.
- **Vantaggi**:
  - Estremamente leggera e veloce, anche su hardware limitato.
  - Adatta a scenari in cui la velocità è critica.
- **Limiti**:
  - Accuratezza significativamente inferiore rispetto a modelli più recenti.
- **Prestazioni**:
  - ~135 FPS su GPU moderne.
  - Funziona bene su dispositivi embedded.
  
---

## Confronto

| **Caratteristica**        | **BiSeNet**           | **BiSeNetV2**        | **ICNet**           | **ENet**            |
|---------------------------|-----------------------|----------------------|---------------------|---------------------|
| **Anno di pubblicazione** | 2018                 | 2020                | 2018                | 2016                |
| **Architettura**          | Dual-path            | Dual-branch ottimizzato | Multi-risoluzione   | Rete compressa      |
| **Velocità (FPS)**        | ~65 FPS              | ~156 FPS            | ~30-60 FPS          | ~135 FPS            |
| **Accuratezza**           | Alta                | Buona               | Moderata            | Bassa               |
| **Risorse richieste**     | Moderate            | Basse               | Molto basse         | Molto basse         |
| **Applicazioni tipiche**  | Generiche            | Dispositivi mobili  | Embedded            | Embedded            |

---

## Conclusioni
- **BiSeNet**: Ideale per chi cerca un buon compromesso tra accuratezza e velocità.
- **BiSeNetV2**: La scelta migliore per real-time segmentation su dispositivi con risorse limitate.
- **ICNet**: Ottima per scenari a risorse estremamente limitate, come i dispositivi mobili.
- **ENet**: La rete più leggera, ma con una precisione ridotta.

La scelta della rete dipende dalle esigenze specifiche dell'applicazione e dalle risorse disponibili.


# SegmentMeIfYouCan: Benchmark per Semantic Segmentation

## Introduzione
**SegmentMeIfYouCan** è un benchmark avanzato per valutare le prestazioni dei modelli di semantic segmentation in contesti reali e sfidanti. A differenza di altri benchmark che si concentrano principalmente su dataset ben definiti e puliti, SegmentMeIfYouCan è progettato per testare la **robustezza**, la **generalizzazione** e l'**efficienza** dei modelli in scenari complessi.

### Obiettivi del Benchmark
- **Robustezza**: Valutare come i modelli affrontano condizioni non ideali, come immagini sfocate, rumore, variazioni di illuminazione e ostruzioni.
- **Efficienza computazionale**: Misurare le prestazioni in termini di velocità di inferenza e utilizzo delle risorse hardware.
- **Generalizzazione**: Testare la capacità dei modelli di segmentare correttamente nuove classi o scenari non presenti nel dataset di addestramento.

---

## Struttura del Benchmark

1. **Dataset diversificati**:
   - SegmentMeIfYouCan utilizza una varietà di dataset per coprire diversi tipi di scenari (urbani, naturali, industriali, ecc.).
   - Include immagini con disturbi come rumore, compressione, e cambiamenti climatici (es. nebbia, pioggia).

2. **Metriche di valutazione**:
   - **mIoU (mean Intersection over Union)**: Accuratezza delle previsioni rispetto al ground truth.
   - **FPS (Frames Per Second)**: Velocità di elaborazione.
   - **Efficienza energetica**: Valutazione dell’uso delle risorse computazionali.
   - **Robustezza**: Prestazioni in condizioni degradate (es. immagini con rumore).

3. **Ambiente di test**:
   - Hardware variegato, dai dispositivi mobili a GPU ad alte prestazioni.
   - Simulazione di condizioni reali per testare la praticità dei modelli.

---

## Collegamento con i Network di Semantic Segmentation

### **BiSeNet**
- **Prestazioni su SegmentMeIfYouCan**:
  - Buona accuratezza su immagini standard grazie al suo approccio bilaterale (Spatial e Context Path).
  - Sensibile a scenari con disturbi pesanti (es. immagini rumorose o sfocate).
- **Punti di forza**:
  - Adatta per dataset complessi ma con condizioni ben controllate.
- **Limiti**:
  - Potrebbe non essere sufficientemente robusta in contesti fortemente degradati.

### **BiSeNetV2**
- **Prestazioni su SegmentMeIfYouCan**:
  - Ottimizzata per velocità, offre risultati real-time anche su hardware limitato.
  - Meno accurata rispetto a BiSeNet in scenari molto complessi o con disturbi pesanti.
- **Punti di forza**:
  - Ideale per test che richiedono alta efficienza computazionale.
- **Limiti**:
  - Il focus sulla velocità sacrifica la robustezza in condizioni estreme.

### **ICNet**
- **Prestazioni su SegmentMeIfYouCan**:
  - Eccellente in termini di efficienza su dispositivi a bassa potenza.
  - Limitazioni significative in accuratezza su dataset altamente disturbati.
- **Punti di forza**:
  - Adatta per test di efficienza energetica e velocità.
- **Limiti**:
  - Non ideale per scenari complessi e con condizioni non ideali.

### **ENet**
- **Prestazioni su SegmentMeIfYouCan**:
  - La rete più leggera, con velocità molto alta anche su hardware molto limitato.
  - Accuratezza inferiore rispetto alle altre reti.
- **Punti di forza**:
  - Perfetta per test che privilegiano la velocità e l’efficienza computazionale.
- **Limiti**:
  - Scarse performance in termini di robustezza e generalizzazione.

---

## Confronto delle Reti su SegmentMeIfYouCan

| **Caratteristica**        | **BiSeNet**           | **BiSeNetV2**        | **ICNet**           | **ENet**            |
|---------------------------|-----------------------|----------------------|---------------------|---------------------|
| **Robustezza**            | Moderata             | Moderata             | Bassa               | Molto bassa         |
| **Velocità**              | Alta (~65 FPS)       | Molto alta (~156 FPS)| Alta (~60 FPS)      | Altissima (~135 FPS)|
| **Generalizzazione**      | Buona                | Discreta             | Bassa               | Molto bassa         |
| **Efficienza hardware**   | Moderata             | Ottima               | Ottima              | Eccellente          |
| **Applicazioni consigliate** | Scenari complessi   | Real-time con hardware limitato | Mobile e embedded | Situazioni con risorse minime |

---

## Conclusioni
SegmentMeIfYouCan evidenzia l'importanza di bilanciare **accuratezza**, **velocità** ed **efficienza computazionale**. Le reti analizzate si distinguono per diverse caratteristiche:
- **BiSeNet**: Adatta per applicazioni che richiedono un compromesso tra accuratezza e velocità.
- **BiSeNetV2**: La scelta migliore per real-time segmentation su dispositivi con risorse limitate.
- **ICNet**: Ideale per dispositivi mobili o scenari con risorse molto limitate.
- **ENet**: Eccellente per velocità ed efficienza, ma non consigliata per scenari complessi.

La scelta del modello dipende fortemente dal contesto applicativo e dai requisiti specifici, come la robustezza alle condizioni avverse e l'hardware disponibile.
