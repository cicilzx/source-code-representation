## Project structure
- AutoenCODE: This folder contains the source code of word2vec.
- graph2vec: This folder contains the source code of graph2vec.
- joern: This folder contains the source code of joern, which is used to extract AST and CFG.
- script: This folder contains the deep learning model to complete the analysis tasks used in our experiment.
- src: This folder contains the code of extracting the syntactic and semantic embedding features of the C++ source code. 

## Usage
### Generating the embedding features
The src/main/java/test/GenerateTrainingData.java file is the entrance of generating embedding feature data set for the C++ source code.
The embedding features of the data set have been stored in the data folder.

### Training the DL/ML model
The script file is the entrance of training different DL/ML models. 