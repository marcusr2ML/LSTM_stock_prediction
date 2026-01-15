# LSTM_stock_prediction
LSTM stock prediction
**This simple script predicts future stock prices of AMAZON based off data from TESLA, MS and APPLE in two ways**
There are two LSTM classes for each method with a bidirectional network being used in both with dropout. A linear layer is also used after the LSTM to make the final predictions.
The number of predictions can be set in the main functions, but I choose to just use 1 being the close price.

1. **LSTM batch**
   This trains one LSTM model on the 3 stocks given above. The data will be batched all at once and the LSTM_time_series.py class will be used. This attempts to learn some cross correlation     between the stocks to better predict AMAZON's

3. **LSTM ensemble**
   This trains three separate LSTM models on the 3 stocks given above. The training is completed one at a time using the LSTM_time_series_ensemble.py class. A weighted average between each      models predicted stock for AMAZON will be included.


The batched data tends to provide better predictive power than the ensemble method, potentially due to a greater varaity of data present in any cross correlations between the stocks. A simple superposition of predictions requires rescalling of each model too, making it more complicated. 

A potential future script could train more stocks as well as various combinations of batched versions of the stocks and return a larger ensemble. This ensemble could also be weighted in a non-uniform way, potentially using a NN to choose weights for a more accurate output. 
