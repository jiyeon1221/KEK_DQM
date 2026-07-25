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
#include "TApplication.h"
#include "TRootCanvas.h"
#include "TPaveStats.h"

#include "TLegend.h"
#include "TLatex.h"
#include "TColor.h"

#include "function.h"

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
	gStyle->SetPalette(kVisibleSpectrum);
	gStyle->SetStatFormat("6.6g");
  
	// setup for prompt analysis
	int fRunNum = std::stoi(argv[1]);
	int fMaxEvent = std::stoi(argv[2]);
	int fStartBin = std::stoi(argv[3]);
	int fEndBin = std::stoi(argv[4]);
	int fMaxFile = -1;

    std::vector<std::string> channel_names;
    std::string full_channel_name = "";
    myColorPalette.clear();
	for (int plot_args = 5; plot_args < argc; plot_args++)
	{
		channel_names.push_back(argv[plot_args]);
		full_channel_name += std::string(argv[plot_args]) + "_";
        myColorPalette.push_back(gStyle->GetColorPalette( (plot_args - 10) * ((float)gStyle->GetNumberOfColors() / ((float)argc - 10.))));
	}
    full_channel_name = full_channel_name.substr(0, full_channel_name.size() - 1);

	fs::path dir("./IntADC");   
	if (!(fs::exists(dir))) fs::create_directory(dir);
    fs::path dir2("./IntADC/Run_" + std::to_string(fRunNum));
    if (!(fs::exists(dir2))) fs::create_directory(dir2);

	// Cut values
	float cut_CC1 = 200.;
	float cut_CC2 = 200.;
	float cut_DWC = 1.5;

	// Integral range
	// int start_bin = 650;
	// int end_bin = 800;
	// height for text legend
	TLatex* text = new TLatex();
	text->SetTextSize(0.025);  
	float height = 0.87;

	// initialize the utility class
	TButility util = TButility();
	util.LoadMapping("../mapping/mapping_KEK.root");
	
	// TFile *f_DWC = TFile::Open((TString)("./DWC/DWC_Run_" + std::to_string(fRunNum) + ".root"), "READ");
	// TH2D *h_DWC1_pos = (TH2D *)f_DWC->Get("dwc1_pos");
	// TH2D *h_DWC2_pos = (TH2D *)f_DWC->Get("dwc2_pos");
	// std::vector<float> DWC1_offset = getDWCoffset(h_DWC1_pos); // DWC1_offset.at(0) == X, DWC1_offset.at(1) == Y
	// std::vector<float> DWC2_offset = getDWCoffset(h_DWC2_pos);


	std::vector<TBcid> cids;
	std::vector<TH1F *> plots;
	for (int idx = 0; idx < channel_names.size(); idx++) {
		plots.push_back(new TH1F((TString)channel_names.at(idx), ";intADC;nEvents", 840, -18000, 350000));
		cids.push_back(util.GetCID(channel_names.at(idx)));
	}

    // MID: 3-7: PMT modules, MID 9: LC, MID 10: Aux(CC1, CC2, PS, TC, MC), MID 12: Triggers (T1, T2, T1NIM, T2NIM, Coin), MID 14-17: MCP micro, MID 18: DWC
    TBread<TBwaveform> readerWave = TBread<TBwaveform>(fRunNum, fMaxEvent, fMaxFile, false, "/pnfs/knu.ac.kr/data/cms/store/user/sungwon/2025_KEK_TB_Data", {8, 9, 13});
    // TBread<TBwaveform> readerWave = TBread<TBwaveform>(fRunNum, fMaxEvent, fMaxFile, false, "/Volumes/Macintosh HD-1/Volumes/HDD_16TB_3", {3, 4, 5, 6, 7, 9, 10, 12, 14, 15, 16, 17, 18});

	// Set Maximum event
    if (fMaxEvent == -1 || fMaxEvent > readerWave.GetMaxEvent())
    fMaxEvent = readerWave.GetMaxEvent();

	TCanvas *c = new TCanvas("c", "c", 1000, 800);
	c->cd();

	TLegend* leg = new TLegend(0.75, 0.2, 0.9, 0.4);

    TFile* outFile = new TFile( ( "./IntADC/Run_" + std::to_string(fRunNum) + "/Run_" + std::to_string(fRunNum) + ".root" ).c_str(), "RECREATE");
    outFile->cd();

    // Starting Evt Loop
    for (int iEvt = 0; iEvt < fMaxEvent; iEvt++) {
		printProgress(iEvt, fMaxEvent);

		// Load event
		TBevt<TBwaveform> aEvent = readerWave.GetAnEvent();
		// filling plots
		for (int idx = 0; idx < plots.size(); idx++)
		{
			auto single_waveform = aEvent.GetData(cids.at(idx)).waveform();

			float IntADC = GetInt(single_waveform, fStartBin, fEndBin);
			plots.at(idx)->Fill(IntADC);
		}
  	} //end of event loop
	
	outFile->cd();
	for (int idx = 0; idx < plots.size(); idx++) {
		plots.at(idx)->Draw("hist");
		plots.at(idx)->Write();
	}
	outFile->Close();

	return 0;
}