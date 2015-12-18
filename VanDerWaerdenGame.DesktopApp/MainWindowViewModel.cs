﻿using Prism.Mvvm;
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

            GameManager.Player1 = PositionPlayers.Last();
            GameManager.Player2 = ColorPlayers.Last();
        }

        public List<IColorPlayer> ColorPlayers { get; set; }
        public List<IPositionPlayer> PositionPlayers { get; set; }

        public GameManager GameManager { get { return gameManager; } set { SetProperty(ref gameManager, value); } }
        private GameManager gameManager = new GameManager(new VanDerWaerdenGameRules());

        public void StartNewGame() { Task.Factory.StartNew(GameManager.NewGame); }
        public void Turn() { Task.Factory.StartNew(GameManager.IterateTurn); }
        public void PlayTillEnd() { Task.Factory.StartNew(() => GameManager.PlayTillEnd(true)); }

        public bool ShouldTrainP1 { get { return shouldTrainP1; } set { SetProperty(ref shouldTrainP1, value); } }
        private bool shouldTrainP1;
        public bool ShouldTrainP2 { get { return shouldTrainP2; } set { SetProperty(ref shouldTrainP2, value); } }
        private bool shouldTrainP2;
        public double P1Efficiency { get { return p1Efficiency; } set { SetProperty(ref p1Efficiency, value); } }
        private double p1Efficiency;
        public double P2Efficiency { get { return p2Efficiency; } set { SetProperty(ref p2Efficiency, value); } }
        private double p2Efficiency;
        public bool IsTraining { get { return isTraining; } set { SetProperty(ref isTraining, value); } }
        private bool isTraining;



        public int NTrainingIterations { get { return nTrainingIterations; } set { SetProperty(ref nTrainingIterations, value); } }
        private int nTrainingIterations = 200;
        public int TrainingIteration { get { return trainingIteration; } set { SetProperty(ref trainingIteration, value); } }
        private int trainingIteration = 0;

        public int NRandomTrainingGames { get { return nRandomTrainingGames; } set { SetProperty(ref nRandomTrainingGames, value); } }
        private int nRandomTrainingGames = 1;



        //Niech gracz uczy się na podstawie N rozgrywek przeciwnika z losowcem?
        //Niech obie sieci dostają wynik rozgrywki... wymaga własnej implementacji.
        public void TrainPlayers()
        {
            IsTraining = true;

            TrainingIteration = 0;

            ITrainable P1 = GameManager.Player1 as ITrainable;
            ITrainable P2 = GameManager.Player2 as ITrainable;
            PositionPlayerTrainer P1Trainer = new PositionPlayerTrainer(GameManager.Rules, gameManager.Player2) { NGames = this.NRandomTrainingGames };
            ColorPlayerTrainer P2Trainer = new ColorPlayerTrainer(GameManager.Rules, gameManager.Player1) { NGames = this.NRandomTrainingGames };
            NeuralSimulatedAnnealing training1 = null, training2 = null;

            if(P1!=null)
                training1 = new NeuralSimulatedAnnealing(P1.Network, P1Trainer, 20, 2, this.NTrainingIterations);
            if(P2!=null)
                training2 = new NeuralSimulatedAnnealing(P2.Network, P2Trainer, 20, 2, this.NTrainingIterations);


            
            for (; TrainingIteration < NTrainingIterations; TrainingIteration++)
            {
                if (ShouldTrainP1 && P1 != null)
                {
                    training1.Iteration();
                }
                if (ShouldTrainP2 && P2 != null)
                {
                    training2.Iteration();
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
            for (int i = 0; i < NTrainingIterations; i++)
            {
                //gm.NewGame();
                gameLengths.Add(gm.PlayGame());
            }
            P1Efficiency = P2Efficiency = gameLengths.Average();
        }
    }
   
}
