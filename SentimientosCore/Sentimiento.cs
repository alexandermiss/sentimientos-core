using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;


namespace SentimientosCore
{
    class Sentimiento
    {
        public class SentimentData
        {
            [Column(ordinal: "0", name: "Label")]
            public int Sentiment;
            [Column(ordinal: "1")]
            public string SentimentText;
        }

        public class SentimentPrediction
        {
            [ColumnName("PredictedLabel")]
            public bool Prediction { get; set; }

            [ColumnName("Probability")]
            public float Probability { get; set; }

            [ColumnName("Score")]
            public float Score { get; set; }
        }
    }
}
