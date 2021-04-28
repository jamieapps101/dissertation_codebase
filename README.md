# Dissertation project work of Jamie Apps

This repo contains the necessary code to run the topic inference NLP project. It consists of 7 dockerised applications:
- speech2text, using Mozilla's DeepSpeech
- text2topic, using a tensorflow model with structure based on [this](https://arxiv.org/abs/2010.03138)
- topic2command, copmaring incoming speech topics to existing command database
- bert_as_service, based on [this](https://github.com/hanxiao/bert-as-service) repo. Required by the text2topic and topic2command modules
- word2vec, using code found [here](https://github.com/jamieapps101/ google_word2vec_rust_inference_server). Required by the text2topic module.
- end2endtest, a module connecting to each of the above modules, allowing end to end testing with submodule evaluation. Also providing the libs required to graph the data.
- mosquitto, as found [here](https://hub.docker.com/_/eclipse-mosquitto), used to connect all the modules together

The control python script, can be run using 
```csh
python3 ./control <function>.<mode>.<container> <ARGS>
```
where function can be one of:
- build
- run
mode can be one of:
- cpu
- gpu
and the container is the name of any of the above modules.
The available args can befound by running 
```sh
python3 ./control -h
```