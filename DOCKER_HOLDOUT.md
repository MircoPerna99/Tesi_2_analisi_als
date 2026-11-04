# Esecuzione offline di Holdout_2

Sulla macchina collegata al database, esporta i dati:

```bash
venv/bin/python export_holdout_data.py
```

La cartella `offline_data` contiene le interazioni ALS, il gold standard DTI,
le strutture e un `config.json` senza credenziali del database. Per aggiornare
lo snapshot, esegui nuovamente l'esportazione. I file pickle devono provenire
da una fonte attendibile.

Costruisci l'immagine nella cartella del progetto:

```bash
docker build -t als-holdout -f Dockerfile .
```

Trasferisci sulla macchina di destinazione l'immagine, la cartella
`offline_data` e, per riutilizzarle, le mappe `.npy` di
`pre_process/distance_map`:

```bash
docker save -o als-holdout.tar als-holdout
# Sulla macchina di destinazione, dopo il trasferimento:
docker load -i als-holdout.tar
```

Sulla macchina di destinazione prepara una cartella `results` e una cartella
`distance_map` contenente le mappe trasferite, poi esegui dalla cartella che
contiene anche `offline_data`:

```bash
mkdir -p results distance_map
docker run --rm \
  --mount type=bind,src="$(pwd)/offline_data",dst=/offline_data,readonly \
  --mount type=bind,src="$(pwd)/results",dst=/results \
  --mount type=bind,src="$(pwd)/distance_map",dst=/results/pre_process/distance_map \
  als-holdout
```

Sul computer di destinazione non serve MySQL. Se mancano mappe, ESM-1b le
costruisce e alla prima esecuzione puo dover scaricare i propri pesi; in tal
caso serve accesso alla rete. I CSV sono salvati in `results` e possono
sovrascrivere file con lo stesso nome. L'analisi globale richiede comunque
molta memoria e tempo.
