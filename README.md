# Zadanie-rekrutacja-Inpost - Pavlo Khrapko

Przeprowadzono analizę i wytrenowano model predykcyjny LGBM w Python za pomocą PyCaret

Główne pliki: prepare_dataset.ipynb, training_models.ipynb
## Użycie
#create a conda environment
* conda create --name yourenvname python=3.8

#activate conda environment
* conda activate yourenvname

#install pycaret
* pip install -r requirements.txt

## Dane
Z danych dimDates_Task użyto:
  dateId,
  dateWeekOfMonth,
  dateQuarter,
  dateIsWeekend,
  dateIsHolidayInd,
  dateWeekDayStartsMonday

I dodano timestamp dla lepszego rozumienia daty przez model


Z danych Posting_Volumes użyto wszystko i dodano VolumeAvgLastXMonths
(VolumeAvgLast3months,
VolumeAvgLast6months,
VolumeAvgLast9months,
VolumeAvgLast12months)

Dodałem to dlatego, że dla każdego klienta jest różny wynik. I przy użyciu modelu trzeba podać te dane dla konkretnego klienta.

Te dane też nie zawierają dane z ostatniego miesięca, bo robimy predykcję na miesiąc/tydzień na przód i nie będziemy mieli tych danych, bo wprowadzana data jest w przyszłości.


Dla temperatury i pogody użyłem wszystkich cech oprócz Nazwy stacji.

Podział train/test: 6560/1224

## Model
Model regresyjny LightGBM był najlepszym spośród testowanych na podstawie miar MAE (~15000), RMSE (~21000) i R2 (~0.94)


## Predykcja i ewaluacja
Ewaluacja była robiona na 1224 rekordach które reprezentują ostatnie 5 miesięcy (od 2023-4-1 do 2023-8-31)
![image](https://github.com/user-attachments/assets/f89a0dd8-f56e-4012-8560-9fc309d89402)
![image](https://github.com/user-attachments/assets/be6feaed-7848-4585-84a1-b2fb97ebba32)

Feature importance:
![image](https://github.com/user-attachments/assets/d3cf82c6-634c-4c29-9a9b-8653b5036404)

