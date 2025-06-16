class NeuralNetwork {
    constructor(layerSizes, learningRate = 0.1) {
        if (!Array.isArray(layerSizes) || layerSizes.length < 2) {
            throw new Error("Layer sizes must be an array with at least an input and output layer.");
        }
        this.layerSizes = layerSizes;
        this.lr = learningRate;
        this.weights = [];
        this.biases = [];
        this.initializeWeights();
    }

    initializeWeights() {
        this.weights = [];
        this.biases = [];
        for (let i = 0; i < this.layerSizes.length - 1; i++) {
            const inputNodes = this.layerSizes[i];
            const outputNodes = this.layerSizes[i + 1];
            
            // Xavier/Glorot initialization for better weight scaling
            const xavier = Math.sqrt(1 / inputNodes);

            const weightMatrix = this.createMatrix(outputNodes, inputNodes)
                .map(row => row.map(() => (Math.random() * 2 - 1) * xavier));
            this.weights.push(weightMatrix);

            const biasMatrix = this.createMatrix(outputNodes, 1)
                .map(() => [Math.random() * 2 - 1]);
            this.biases.push(biasMatrix);
        }
    }

    // --- ACTIVATION FUNCTIONS ---
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    softmax(arr) {
        const maxLogit = Math.max(...arr); // For numerical stability
        const exps = arr.map(x => Math.exp(x - maxLogit));
        const sumExps = exps.reduce((a, b) => a + b);
        return exps.map(e => e / sumExps);
    }

    // --- MATRIX UTILITY FUNCTIONS ---
    createMatrix(rows, cols) {
        return Array(rows).fill().map(() => Array(cols).fill(0));
    }

    fromArray(arr) {
        return arr.map(val => [val]);
    }

    toArray(matrix) {
        return matrix.flat();
    }
    
    multiply(a, b) {
        if (a[0].length !== b.length) {
            console.error("Matrix dimensions are not compatible for multiplication.");
            return;
        }
        let result = this.createMatrix(a.length, b[0].length);
        for (let i = 0; i < result.length; i++) {
            for (let j = 0; j < result[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < a[0].length; k++) {
                    sum += a[i][k] * b[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    add(a, b) {
        return a.map((row, i) => row.map((val, j) => val + b[i][j]));
    }
    
    map(matrix, func) {
        return matrix.map((row, i) => row.map((val, j) => func(val, i, j)));
    }

    // --- FORWARD PROPAGATION ---
    predict(input_array) {
        let current_layer_output = this.fromArray(input_array);

        // Process through each layer
        for (let i = 0; i < this.weights.length; i++) {
            current_layer_output = this.multiply(this.weights[i], current_layer_output);
            current_layer_output = this.add(current_layer_output, this.biases[i]);
            
            // Apply sigmoid to all hidden layers, but not the final output layer
            if (i < this.weights.length - 1) {
                current_layer_output = this.map(current_layer_output, this.sigmoid);
            }
        }
        
        // Apply softmax to the final output layer to get probabilities
        return this.softmax(this.toArray(current_layer_output));
    }
    
    // --- WEIGHTS MANAGEMENT ---
    loadWeights(data) {
        // The GPU-trained weights are a JSON object with a "weights" key,
        // containing the array of tensor objects. Check for this format first.
        if (data && data.weights && !data.structure) {
            const tfTensors = data.weights;

            // Verify if it looks like a TF tensor array before proceeding
            if (Array.isArray(tfTensors) && tfTensors.length > 0 && tfTensors[0]?.data && tfTensors[0]?.shape) {
                console.log("TensorFlow.js weights format detected. Converting...");
                
                const newWeights = [];
                const newBiases = [];

                if (tfTensors.length !== (this.layerSizes.length - 1) * 2) {
                    throw new Error(`Incompatible number of weight/bias tensors. Expected ${(this.layerSizes.length - 1) * 2}, got ${tfTensors.length}`);
                }

                for(let i = 0; i < tfTensors.length; i += 2) {
                    const weightTensor = tfTensors[i];
                    const biasTensor = tfTensors[i+1];
                    
                    // 1. Process Weights: Reshape and Transpose from [in, out] to [out, in]
                    const [rows, cols] = weightTensor.shape;
                    const flatWeights = weightTensor.data;
                    const transposedWeights = this.createMatrix(cols, rows); 
                    for (let r = 0; r < rows; r++) {
                        for (let c = 0; c < cols; c++) {
                            transposedWeights[c][r] = flatWeights[r * cols + c];
                        }
                    }
                    newWeights.push(transposedWeights);

                    // 2. Process Biases: Convert flat array to a column matrix
                    const biasMatrix = this.fromArray(biasTensor.data);
                    newBiases.push(biasMatrix);
                }

                this.weights = newWeights;
                this.biases = newBiases;
                console.log("Successfully loaded and converted TensorFlow.js weights.");
                return; // Exit after successful conversion
            }
        }
        
        // --- Handle Original/Old Formats ---
        if (!data || !data.structure) {
            throw new Error("Invalid weights file format: Missing 'structure' key or unrecognized format.");
        }

        let fileLayers;
        let fileWeights = [];
        let fileBiases = [];

        // Check if it's the NEW format (structure is an array)
        if (Array.isArray(data.structure)) {
            fileLayers = data.structure;
            fileWeights = data.weights;
            fileBiases = data.biases;
            if (!fileWeights || !fileBiases) {
                 throw new Error("Invalid weights file format: Missing 'weights' or 'biases' keys for new format.");
            }
        } 
        // Check if it's an OLD format (structure is an object)
        else if (typeof data.structure === 'object') {
            console.log("Old weight format detected. Attempting to convert...");
            const s = data.structure;
            // 1-layer
            if (s.hiddenNodes) {
                fileLayers = [s.inputNodes, s.hiddenNodes, s.outputNodes];
                fileWeights = [data.weights_ih, data.weights_ho];
                fileBiases = [data.bias_h, data.bias_o];
            } 
            // 2-layer
            else if (s.hiddenNodes1 && s.hiddenNodes2 && !s.hiddenNodes3) {
                fileLayers = [s.inputNodes, s.hiddenNodes1, s.hiddenNodes2, s.outputNodes];
                fileWeights = [data.weights_ih1, data.weights_h1h2, data.weights_h2o];
                fileBiases = [data.bias_h1, data.bias_h2, data.bias_o];
            }
            // 3-layer
            else if (s.hiddenNodes3) {
                fileLayers = [s.inputNodes, s.hiddenNodes1, s.hiddenNodes2, s.hiddenNodes3, s.outputNodes];
                fileWeights = [data.weights_i_h1, data.weights_h1_h2, data.weights_h2_h3, data.weights_h3_o];
                fileBiases = [data.bias_h1, data.bias_h2, data.bias_h3, data.bias_o];
            }
            
            if (fileWeights.includes(undefined) || fileBiases.includes(undefined)) {
                 throw new Error("Old weights file is corrupted or has missing keys.");
            }
        }
        else {
            throw new Error("Unrecognized weights file format.");
        }
        
        // Final validation: Compare the architecture from the file with the current network instance
        if (JSON.stringify(fileLayers) !== JSON.stringify(this.layerSizes)) {
            throw new Error(`Weight file architecture mismatch. Expected [${this.layerSizes.join(', ')}] but file is for [${fileLayers.join(', ')}].`);
        }

        this.weights = fileWeights;
        this.biases = fileBiases;
        console.log("Successfully loaded and validated weights.");
    }
}
