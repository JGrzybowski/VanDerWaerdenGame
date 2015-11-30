﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace VanDerWaerdenGame.DesktopApp
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            
        }

        public MainWindowViewModel ViewModel => this.DataContext as MainWindowViewModel;

        private void StartGameButton_Click(object sender, RoutedEventArgs e)
        {
            ViewModel.StartNewGame();
        }
            
        private void NextTurnButton_Click(object sender, RoutedEventArgs e)
        {
            Task.Factory.StartNew(ViewModel.Turn);
        }
        
        private void PlayTillEndButton_Click(object sender, RoutedEventArgs e)
        {
            Task.Factory.StartNew(ViewModel.PlayTillEnd);
        }
    }
}
