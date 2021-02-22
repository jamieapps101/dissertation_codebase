import model_generation
import model_inference
import model_training


if __name__=="__main__":
    model=model_generation.build_Att_BiLSTM()
    print(model.summary())