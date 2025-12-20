from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell

class model_configuration:

    def compile_and_fit(model, window, patience=3, max_epochs=50):
        """
        function that configures the model for training and then fits the model on the data, as 
        shown in the following listing.
        
        :param model: Model to be compiled and trained.
        :param window: An instance of the DataWindow class containing training and validation data.
        :param patience: Number of epochs with no improvement after which training will be stopped.
        :param max_epochs: sets amaximum number of epochs to train the model.
        """
        
        # Early stopping occurs if 3 consecutive epochs do not decrease 
        # # the validation loss, as set by the patience parameter.
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=patience,
                                       mode='min')
        
        # The MSE is used as the loss function.
        model.compile(loss=MeanSquaredError(),
                      optimizer=Adam(),
                      metrics=[MeanAbsoluteError()])
        
        # The model is fit using the training data from the window object.
        history = model.fit(window.train,
                            epochs=max_epochs,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        
        return history
    




