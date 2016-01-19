using Prism.Mvvm;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using VanDerWaerdenGame.Model;
using VanDerWaerdenGame.Players.ColorChoosers;
using VanDerWaerdenGame.Players.PositionChoosers;
using VanDerWaerdenGame.Players;
using Encog.Neural.Networks.Training.Anneal;
using System.IO;

namespace VanDerWaerdenGame.DesktopApp
{
    public class MainWindowViewModel : BindableBase
    {
        public MainWindowViewModel() : base()
        {
            //Player is loaded just to have static linking ot the Players assembly
            var plr = new RandomColorPlayer(GameManager.Rules as VanDerWaerdenGameRules);

            ColorPlayers = AppDomain.CurrentDomain.GetAssemblies()
                        .SelectMany(s => s.GetTypes())
                        .Where(p => typeof(IColorPlayer).IsAssignableFrom(p) && p.IsClass && !p.IsAbstract)
                        .Select(t => (IColorPlayer)Activator.CreateInstance(t, GameManager.Rules))
                        .ToList();

            PositionPlayers = AppDomain.CurrentDomain.GetAssemblies()
                        .SelectMany(s => s.GetTypes())
                        .Where(p => typeof(IPositionPlayer).IsAssignableFrom(p) && p.IsClass && !p.IsAbstract)
                        .Select(t => (IPositionPlayer)Activator.CreateInstance(t, GameManager.Rules))
                        .ToList();

            foreach (PlayerBase player in ColorPlayers)
                player.Rules = gameManager.Rules as VanDerWaerdenGameRules;
            foreach (PlayerBase player in ColorPlayers)
                player.Rules = gameManager.Rules as VanDerWaerdenGameRules;

            GameManager.Player1 = PositionPlayers.First();
            GameManager.Player2 = ColorPlayers.Last();

            string[] args = Environment.GetCommandLineArgs();
            if (args.Count() > 1)
                RunForCommandLineParams(args);
        }

        private void RunForCommandLineParams(string[] args)
        {
            var parameters = new Dictionary<string, string>();
            for (int index = 1; index < args.Length; index += 2)
                parameters.Add(args[index], args[index + 1]);

            var rules = this.GameManager.Rules as VanDerWaerdenGameRules;
            rules.EndGameProgressionLength = int.Parse(parameters["k"]);
            rules.NColors = int.Parse(parameters["r"]);

            this.StepSize = double.Parse(parameters["step"]);
            this.NStepGames = int.Parse(parameters["itperstep"]);

            this.TMax = int.Parse(parameters["tmax"]);
            this.TMin = int.Parse(parameters["tmin"]);
            try
            {
                TrainPositionPlayer();

                if (parameters["opponent"].Equals("mean", StringComparison.InvariantCultureIgnoreCase) || parameters["opponent"].Equals("both", StringComparison.InvariantCultureIgnoreCase))
                {
                    this.GameManager.Player2 = new MeanColorPlayer(rules);
                    TestPlayers(string.Format("r{0}_k{1}_step{2:00000}_itperstep{3:0000}_tmax{4:000}_tmin{5:000}_{6}{7}.csv",
                        rules.NColors, rules.EndGameProgressionLength,
                        this.StepSize * 10000, this.NStepGames,
                        this.TMax, this.TMin,
                        "mean",
                        parameters.ContainsKey("try") ? parameters["try"] : ""
                        ));
                }
                if (parameters["opponent"].Equals("rand", StringComparison.InvariantCultureIgnoreCase) || parameters["opponent"].Equals("both", StringComparison.InvariantCultureIgnoreCase))
                {
                    this.GameManager.Player2 = new RandomColorPlayer(rules);
                    TestPlayers(string.Format("r{0}_k{1}_step{2:00000}_itperstep{3:0000}_tmax{4:000}_tmin{5:000}_{6}{7}.csv",
                        rules.NColors, rules.EndGameProgressionLength,
                        this.StepSize * 10000, this.NStepGames,
                        this.TMax, this.TMin,
                        "rand",
                        parameters.ContainsKey("try") ? parameters["try"] : ""
                        ));
                }
                else
                    throw new ArgumentException("You have not provided proper value of opponent paramerter (mean/rand/both)");
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
                Console.WriteLine(e.StackTrace);
            }
            finally
            {
                App.Current.Shutdown();
            }
        }

        public List<IColorPlayer> ColorPlayers { get; set; }
        public List<IPositionPlayer> PositionPlayers { get; set; }

        public GameManager GameManager { get { return gameManager; } set { SetProperty(ref gameManager, value); } }
        private GameManager gameManager = new GameManager(new VanDerWaerdenGameRules());

        public async Task StartNewGame() { await Task.Factory.StartNew(GameManager.NewGame); }
        public async Task Turn() { await Task.Factory.StartNew(GameManager.IterateTurn); }
        public async Task PlayTillEnd() { await Task.Factory.StartNew(() => GameManager.PlayTillEnd(true)); }

        public bool ShouldTrainP1 { get { return shouldTrainP1; } set { SetProperty(ref shouldTrainP1, value); } }
        private bool shouldTrainP1 = true;
        //public bool ShouldTrainP2 { get { return shouldTrainP2; } set { SetProperty(ref shouldTrainP2, value); } }
        //private bool shouldTrainP2;
        public double P1Efficiency { get { return p1Efficiency; } set { SetProperty(ref p1Efficiency, value); } }
        private double p1Efficiency;
        //public double P2Efficiency { get { return p2Efficiency; } set { SetProperty(ref p2Efficiency, value); } }
        //private double p2Efficiency;
        public bool IsTraining { get { return isTraining; } set { SetProperty(ref isTraining, value); } }
        private bool isTraining;
        
        public int NTestingIterations { get { return nTrainingIterations; } set { SetProperty(ref nTrainingIterations, value); } }
        private int nTrainingIterations = 5000;
        public int TrainingIteration { get { return trainingIteration; } set { SetProperty(ref trainingIteration, value); } }
        private int trainingIteration = 0;

        public double StepSize { get { return stepSize; } set { SetProperty(ref stepSize, value); } }
        private double stepSize = 0.05;
        public int NStepGames { get { return nStepGames; } set { SetProperty(ref nStepGames, value); } }
        private int nStepGames = 35;

        public int TMax { get { return tmax; } set { SetProperty(ref tmax, value); } }
        private int tmax = 20;
        public int TMin { get { return tmin; } set { SetProperty(ref tmin, value); } }
        private int tmin = 2;

        public void TrainPositionPlayer()
        {
            IsTraining = true;

            TrainingIteration = 0;
            var nSteps = (int)Math.Ceiling(1.0 / StepSize);

            ITrainable P1 = GameManager.Player1 as ITrainable;
            var P1Trainer = new PositionStepTrainer(GameManager.Rules);
            NeuralSimulatedAnnealing training = null;

            if (P1 != null)
            {
                training = new NeuralSimulatedAnnealing(P1.Network, P1Trainer, TMax, TMin, nSteps * NStepGames);

                if (ShouldTrainP1)
                    for (int i = 0; i < nSteps; i++)
                    {
                        P1Trainer.Distortion = 1 - i * StepSize;
                        for (int j = 0; j < NStepGames; j++)
                        {
                            training.Iteration();
                            TrainingIteration++;
                        }
                    }
            }
            IsTraining = false;
        }

        public void TestPlayers(string fileName = null)
        {
            var gm = new GameManager(this.GameManager.Rules) {
                Player1 = GameManager.Player1,
                Player2 = GameManager.Player2,
            };
            if (fileName != null)
                gm.Logger = new AppendGameLogger(fileName);

            List<int> gameLengths = new List<int>();
            for (int i = 0; i < NTestingIterations; i++)
            {
                //gm.NewGame();
                gameLengths.Add(gm.PlayGame());
            }
            P1Efficiency = gameLengths.Average();
        }
    }
   
}
