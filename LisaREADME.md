# Ambiente virtuale 
## Creazione
Sposto il terminale nella cartella del progetto
	cd ~/Desktop/AML-AnomalySegmentation

Creo l'ambiente virtuale
	python3 -m venv myenv

Lo attivo
	source myenv/bin/activate

Installo i pacchetti
1.	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu <- Non posso usare CUDA perchè ho M1 quindi uso Metal
2.	pip install numpy matplotlib Pillow visdom

## Attivare l'ambiente virtuale
Ogni volta che voglio lavorare nell'ambiente posso attivarlo con:
	source myenv/bin/activate