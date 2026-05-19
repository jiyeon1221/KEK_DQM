#include "TBread.h"
#include "TButility.h"

#include <filesystem>
#include <chrono>
#include <numeric>
#include <vector>
#include "stdlib.h"
#include "stdio.h"
#include "string.h"

#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TF1.h"

#include "function.h"

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {

    // setup for prompt analysis
    // it could be like
    int fRunNum = std::stoi(argv[1]);
    int fMaxEvent = std::stoi(argv[2]);
    
    fs::path dir("./Hodoscope");   
    if (!(fs::exists(dir))) fs::create_directory(dir);
        
    // initialize the utility class
    TButility util = TButility();
    util.LoadMapping("../mapping/mapping_KEK.root");

    // TODO: Update integration range, or use peakADC instead of intADC
    int first = 135;  // Hodoscope integration range
    int last  = 270;  // Hodoscope integration range
    
    // 16x16 fiber CID
    // TBcid cid_X1  = util.GetCID("X1"); 
    // TBcid cid_X2  = util.GetCID("X2"); 
    // TBcid cid_X3  = util.GetCID("X3"); 
    // TBcid cid_X4  = util.GetCID("X4"); 
    // TBcid cid_X5  = util.GetCID("X5"); 
    // TBcid cid_X6  = util.GetCID("X6"); 
    // TBcid cid_X7  = util.GetCID("X7"); 
    // TBcid cid_X8  = util.GetCID("X8"); 
    // TBcid cid_X9  = util.GetCID("X9"); 
    // TBcid cid_X10 = util.GetCID("X10");
    // TBcid cid_X11 = util.GetCID("X11");
    // TBcid cid_X12 = util.GetCID("X12");
    // TBcid cid_X13 = util.GetCID("X13");
    // TBcid cid_X14 = util.GetCID("X14");
    // TBcid cid_X15 = util.GetCID("X15");
    // TBcid cid_X16 = util.GetCID("X16");
    
    // TBcid cid_Y1  = util.GetCID("Y1");
    // TBcid cid_Y2  = util.GetCID("Y2");
    // TBcid cid_Y3  = util.GetCID("Y3");
    // TBcid cid_Y4  = util.GetCID("Y4");
    // TBcid cid_Y5  = util.GetCID("Y5");
    // TBcid cid_Y6  = util.GetCID("Y6");
    // TBcid cid_Y7  = util.GetCID("Y7");
    // TBcid cid_Y8  = util.GetCID("Y8");
    // TBcid cid_Y9  = util.GetCID("Y9");
    // TBcid cid_Y10 = util.GetCID("Y10");
    // TBcid cid_Y11 = util.GetCID("Y11");
    // TBcid cid_Y12 = util.GetCID("Y12");
    // TBcid cid_Y13 = util.GetCID("Y13");
    // TBcid cid_Y14 = util.GetCID("Y14");
    // TBcid cid_Y15 = util.GetCID("Y15");
    // TBcid cid_Y16 = util.GetCID("Y16");

    TBcid cid_X1  = util.GetCID("HX1");
    TBcid cid_X2  = util.GetCID("HX2");
    TBcid cid_X3  = util.GetCID("HX3");
    TBcid cid_X4  = util.GetCID("HX4");
    TBcid cid_X5  = util.GetCID("HX5");
    TBcid cid_X6  = util.GetCID("HX6");
    TBcid cid_X7  = util.GetCID("HX7");
    TBcid cid_X8  = util.GetCID("HX8");
    TBcid cid_X9  = util.GetCID("HX9");
    TBcid cid_X10 = util.GetCID("HX10");
    TBcid cid_X11 = util.GetCID("HX11");
    TBcid cid_X12 = util.GetCID("HX12");
    TBcid cid_X13 = util.GetCID("HX13");
    TBcid cid_X14 = util.GetCID("HX14");
    TBcid cid_X15 = util.GetCID("HX15");
    TBcid cid_X16 = util.GetCID("HX16");

    TBcid cid_Y1  = util.GetCID("HY1");
    TBcid cid_Y2  = util.GetCID("HY2");
    TBcid cid_Y3  = util.GetCID("HY3");
    TBcid cid_Y4  = util.GetCID("HY4");
    TBcid cid_Y5  = util.GetCID("HY5");
    TBcid cid_Y6  = util.GetCID("HY6");
    TBcid cid_Y7  = util.GetCID("HY7");
    TBcid cid_Y8  = util.GetCID("HY8");
    TBcid cid_Y9  = util.GetCID("HY9");
    TBcid cid_Y10 = util.GetCID("HY10");
    TBcid cid_Y11 = util.GetCID("HY11");
    TBcid cid_Y12 = util.GetCID("HY12");
    TBcid cid_Y13 = util.GetCID("HY13");
    TBcid cid_Y14 = util.GetCID("HY14");
    TBcid cid_Y15 = util.GetCID("HY15");
    TBcid cid_Y16 = util.GetCID("HY16");

    // prepare the histograms wa want to draw
    TH2F* hist_hodoscope_intADC = new TH2F("hodoscope_intADC" , "Hodoscope IntADC;X[mm];Y[mm];events", 16, 0, 16, 16, 0, 16);
    TH2F* hist_hodoscope_peakADC = new TH2F("hodoscope_peakADC" , "Hodoscope PeakADC;X[mm];Y[mm];events", 16, 0, 16, 16, 0, 16);

    // Preapare data reader
    // TODO: Update MID to proper DAQ number
    TBread<TBwaveform> readerWave = TBread<TBwaveform>(fRunNum, fMaxEvent, -1, false, "/u/user/swkim/SE_UserHome/2025_KEK_TB_Data", {8, 9});
    
    // Set Maximum event
    if (fMaxEvent == -1 || fMaxEvent > readerWave.GetMaxEvent())
        fMaxEvent = readerWave.GetMaxEvent();
    
    for (int iEvt = 0; iEvt < fMaxEvent; iEvt++) {
        if (iEvt % 100 == 0) printProgress(iEvt, fMaxEvent);
        
        // Load event
        TBevt<TBwaveform> anEvt = readerWave.GetAnEvent();
        
        //////////////////////////////////////////////////////////////////////
        // Preparing the waveform for each channel
        //////////////////////////////////////////////////////////////////////
        std::vector<short> wave_X1 = anEvt.GetData(cid_X1).waveform();
        std::vector<short> wave_X2 = anEvt.GetData(cid_X2).waveform();
        std::vector<short> wave_X3 = anEvt.GetData(cid_X3).waveform();
        std::vector<short> wave_X4 = anEvt.GetData(cid_X4).waveform();
        std::vector<short> wave_X5 = anEvt.GetData(cid_X5).waveform();
        std::vector<short> wave_X6 = anEvt.GetData(cid_X6).waveform();
        std::vector<short> wave_X7 = anEvt.GetData(cid_X7).waveform();
        std::vector<short> wave_X8 = anEvt.GetData(cid_X8).waveform();
        std::vector<short> wave_X9 = anEvt.GetData(cid_X9).waveform();
        std::vector<short> wave_X10 = anEvt.GetData(cid_X10).waveform();
        std::vector<short> wave_X11 = anEvt.GetData(cid_X11).waveform();
        std::vector<short> wave_X12 = anEvt.GetData(cid_X12).waveform();
        std::vector<short> wave_X13 = anEvt.GetData(cid_X13).waveform();
        std::vector<short> wave_X14 = anEvt.GetData(cid_X14).waveform();
        std::vector<short> wave_X15 = anEvt.GetData(cid_X15).waveform();
        std::vector<short> wave_X16 = anEvt.GetData(cid_X16).waveform();

        std::vector<short> wave_Y1 = anEvt.GetData(cid_Y1).waveform();
        std::vector<short> wave_Y2 = anEvt.GetData(cid_Y2).waveform();
        std::vector<short> wave_Y3 = anEvt.GetData(cid_Y3).waveform();
        std::vector<short> wave_Y4 = anEvt.GetData(cid_Y4).waveform();
        std::vector<short> wave_Y5 = anEvt.GetData(cid_Y5).waveform();
        std::vector<short> wave_Y6 = anEvt.GetData(cid_Y6).waveform();
        std::vector<short> wave_Y7 = anEvt.GetData(cid_Y7).waveform();
        std::vector<short> wave_Y8 = anEvt.GetData(cid_Y8).waveform();
        std::vector<short> wave_Y9 = anEvt.GetData(cid_Y9).waveform();
        std::vector<short> wave_Y10 = anEvt.GetData(cid_Y10).waveform();
        std::vector<short> wave_Y11 = anEvt.GetData(cid_Y11).waveform();
        std::vector<short> wave_Y12 = anEvt.GetData(cid_Y12).waveform();
        std::vector<short> wave_Y13 = anEvt.GetData(cid_Y13).waveform();
        std::vector<short> wave_Y14 = anEvt.GetData(cid_Y14).waveform();
        std::vector<short> wave_Y15 = anEvt.GetData(cid_Y15).waveform();
        std::vector<short> wave_Y16 = anEvt.GetData(cid_Y16).waveform();
      
        //////////////////////////////////////////////////////////////////////
        // IntADC, PeakADC
        //////////////////////////////////////////////////////////////////////
        std::vector<float> intADC_X(16);
        std::vector<float> intADC_Y(16);
        std::vector<float> peakADC_X(16);
        std::vector<float> peakADC_Y(16);

        intADC_X[0]  = GetInt(wave_X1, first, last);
        intADC_X[1]  = GetInt(wave_X2, first, last);
        intADC_X[2]  = GetInt(wave_X3, first, last);
        intADC_X[3]  = GetInt(wave_X4, first, last);
        intADC_X[4]  = GetInt(wave_X5, first, last);
        intADC_X[5]  = GetInt(wave_X6, first, last);
        intADC_X[6]  = GetInt(wave_X7, first, last);
        intADC_X[7]  = GetInt(wave_X8, first, last);
        intADC_X[8]  = GetInt(wave_X9, first, last);
        intADC_X[9]  = GetInt(wave_X10, first, last);
        intADC_X[10] = GetInt(wave_X11, first, last);
        intADC_X[11] = GetInt(wave_X12, first, last);
        intADC_X[12] = GetInt(wave_X13, first, last);
        intADC_X[13] = GetInt(wave_X14, first, last);
        intADC_X[14] = GetInt(wave_X15, first, last);
        intADC_X[15] = GetInt(wave_X16, first, last);

        intADC_Y[0]  = GetInt(wave_Y1, first, last);
        intADC_Y[1]  = GetInt(wave_Y2, first, last);
        intADC_Y[2]  = GetInt(wave_Y3, first, last);
        intADC_Y[3]  = GetInt(wave_Y4, first, last);
        intADC_Y[4]  = GetInt(wave_Y5, first, last);
        intADC_Y[5]  = GetInt(wave_Y6, first, last);
        intADC_Y[6]  = GetInt(wave_Y7, first, last);
        intADC_Y[7]  = GetInt(wave_Y8, first, last);
        intADC_Y[8]  = GetInt(wave_Y9, first, last);
        intADC_Y[9]  = GetInt(wave_Y10, first, last);
        intADC_Y[10] = GetInt(wave_Y11, first, last);
        intADC_Y[11] = GetInt(wave_Y12, first, last);
        intADC_Y[12] = GetInt(wave_Y13, first, last);
        intADC_Y[13] = GetInt(wave_Y14, first, last);
        intADC_Y[14] = GetInt(wave_Y15, first, last);
        intADC_Y[15] = GetInt(wave_Y16, first, last);

        peakADC_X[0]  = GetPeak(wave_X1, first, last);
        peakADC_X[1]  = GetPeak(wave_X2, first, last);
        peakADC_X[2]  = GetPeak(wave_X3, first, last);
        peakADC_X[3]  = GetPeak(wave_X4, first, last);
        peakADC_X[4]  = GetPeak(wave_X5, first, last);
        peakADC_X[5]  = GetPeak(wave_X6, first, last);
        peakADC_X[6]  = GetPeak(wave_X7, first, last);
        peakADC_X[7]  = GetPeak(wave_X8, first, last);
        peakADC_X[8]  = GetPeak(wave_X9, first, last);
        peakADC_X[9]  = GetPeak(wave_X10, first, last);
        peakADC_X[10] = GetPeak(wave_X11, first, last);
        peakADC_X[11] = GetPeak(wave_X12, first, last);
        peakADC_X[12] = GetPeak(wave_X13, first, last);
        peakADC_X[13] = GetPeak(wave_X14, first, last);
        peakADC_X[14] = GetPeak(wave_X15, first, last);
        peakADC_X[15] = GetPeak(wave_X16, first, last);

        peakADC_Y[0]  = GetPeak(wave_Y1, first, last);
        peakADC_Y[1]  = GetPeak(wave_Y2, first, last);
        peakADC_Y[2]  = GetPeak(wave_Y3, first, last);
        peakADC_Y[3]  = GetPeak(wave_Y4, first, last);
        peakADC_Y[4]  = GetPeak(wave_Y5, first, last);
        peakADC_Y[5]  = GetPeak(wave_Y6, first, last);
        peakADC_Y[6]  = GetPeak(wave_Y7, first, last);
        peakADC_Y[7]  = GetPeak(wave_Y8, first, last);
        peakADC_Y[8]  = GetPeak(wave_Y9, first, last);
        peakADC_Y[9]  = GetPeak(wave_Y10, first, last);
        peakADC_Y[10] = GetPeak(wave_Y11, first, last);
        peakADC_Y[11] = GetPeak(wave_Y12, first, last);
        peakADC_Y[12] = GetPeak(wave_Y13, first, last);
        peakADC_Y[13] = GetPeak(wave_Y14, first, last);
        peakADC_Y[14] = GetPeak(wave_Y15, first, last);
        peakADC_Y[15] = GetPeak(wave_Y16, first, last);


        //////////////////////////////////////////////////////////////////////
        // Find max X and Y position of intADC, peakADC
        //////////////////////////////////////////////////////////////////////
        // float max_X_intADC = *std::max_element(intADC_X.begin(), intADC_X.end());
        // float max_Y_intADC = *std::max_element(intADC_Y.begin(), intADC_Y.end());

        int max_X_idx_intADC = std::max_element(intADC_X.begin(), intADC_X.end()) - intADC_X.begin();
        int max_Y_idx_intADC = std::max_element(intADC_Y.begin(), intADC_Y.end()) - intADC_Y.begin();

        float max_X_pos_intADC = max_X_idx_intADC + 0.5;
        float max_Y_pos_intADC = max_Y_idx_intADC + 0.5;

        // float max_X_peakADC = *std::max_element(peakADC_X.begin(), peakADC_X.end());
        // float max_Y_peakADC = *std::max_element(peakADC_Y.begin(), peakADC_Y.end());

        int max_X_idx_peakADC = std::max_element(peakADC_X.begin(), peakADC_X.end()) - peakADC_X.begin();
        int max_Y_idx_peakADC = std::max_element(peakADC_Y.begin(), peakADC_Y.end()) - peakADC_Y.begin();

        float max_X_pos_peakADC = max_X_idx_peakADC + 0.5;
        float max_Y_pos_peakADC = max_Y_idx_peakADC + 0.5;

        //////////////////////////////////////////////////////////////////////
        // Filling histograms before event selection
        //////////////////////////////////////////////////////////////////////
        hist_hodoscope_intADC->Fill(max_X_pos_intADC, max_Y_pos_intADC, 1);
        hist_hodoscope_peakADC->Fill(max_X_pos_peakADC, max_Y_pos_peakADC, 1);
    }

    //////////////////////////////////////////////////////////////////////
    // Output file
    //////////////////////////////////////////////////////////////////////
    std::string outFile = "./Hodoscope/Hodoscope_Run_" + std::to_string(fRunNum) + ".root";
    TFile* outputRoot = new TFile(outFile.c_str(), "RECREATE");
    outputRoot->cd();
    
    hist_hodoscope_intADC->Write();
    hist_hodoscope_peakADC->Write();

    outputRoot->Close();
}