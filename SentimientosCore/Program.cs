using System;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Learners;
using System.IO;
using static SentimientosCore.Sentimiento;


namespace SentimientosCore
{
    class Program
    {
        static readonly string _rutaDatosEntrenamiento = Path.Combine(Environment.CurrentDirectory, "Data", @"..\..\sentiment labelled sentences\imdb_labelled.txt");
        static readonly string _rutaDatosPrueba = Path.Combine(Environment.CurrentDirectory, "Data", @"..\..\sentiment labelled sentences\yelp_labelled.txt");

        static TextLoader _textLoader;

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            _textLoader = mlContext.Data.CreateTextReader(new TextLoader.Arguments() {
                Separator = "tab",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.Bool, 1),
                    new TextLoader.Column("SentimentText", DataKind.Text, 0)
                }
            });

            var data = _textLoader.Read(_rutaDatosEntrenamiento);
            var pipeline = mlContext.Transforms.Text.FeaturizeText("SentimentText", "Features")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 20));

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = pipeline.Fit(data);

            var predictionEngine = model.CreatePredictionEngine<SentimentData, SentimentPrediction>(mlContext);
            var prediction = predictionEngine.Predict(new SentimentData
            {
                SentimentText = "You are a bad person!"
            });

            Console.WriteLine("prediction: " + prediction.Prediction);
            Console.ReadKey();

        }


    }
}
