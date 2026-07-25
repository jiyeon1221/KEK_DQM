#include "TBread.h"
#include "TButility.h"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <vector>

#include "TCanvas.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TLegend.h"
#include "TROOT.h"
#include "TStyle.h"

#include "function.h"

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    // Set styles
    // For avg time struc, do not need stat box
    gStyle->SetPalette(kVisibleSpectrum);
    gStyle->SetOptStat(0);
    gStyle->SetStatFormat("6.6g");
    
    // setup for prompt analysis
    int fRunNum = std::stoi(argv[1]);
    int fMaxEvent = std::stoi(argv[2]);
    int fMaxFile = -1;    
    std::vector<std::string> channel_names;
    std::string full_channel_name = "";
    // Rescale the line-color palette to the actual number of channels (the
    // default myColorPalette in function.h only has 9 entries and would
    // throw std::out_of_range past that -- same fix as draw_wave.cc /
    // draw_peakADC.cc for many-channel overlays like the full hodoscope).
    myColorPalette.clear();
    for (int plot_args = 3; plot_args < argc; plot_args++) {
        channel_names.push_back(argv[plot_args]);
        full_channel_name += std::string(argv[plot_args]) + "_";
        myColorPalette.push_back(gStyle->GetColorPalette(
            (plot_args - 3) * ((float)gStyle->GetNumberOfColors() / ((float)argc - 3.))));
    }
    full_channel_name = full_channel_name.substr(0, full_channel_name.size() - 1);
    // Guard against filesystem-unfriendly filenames when overlaying a large
    // number of channels (e.g. all 122 hodoscope fibers at once). Keep the
    // first/last channel name in the fallback so different channel sets of
    // the same size (e.g. SHX1..29 vs SHY1..29) don't collide and silently
    // overwrite each other's output.
    if (full_channel_name.size() > 150 || channel_names.size() > 20) {
        full_channel_name = channel_names.front() + "_to_" + channel_names.back()
            + "_" + std::to_string(channel_names.size()) + "ch";
    }
    const bool drawLegend = (channel_names.size() <= 20);
    
    // Create output directory
    fs::path dir("./Avg");
    if (!(fs::exists(dir))) fs::create_directory(dir);
    fs::path dir2("./Avg/Run_" + std::to_string(fRunNum));
    if (!(fs::exists(dir2))) fs::create_directory(dir2);
    
    // Load mapping
    // Test mapping derived from test_Hodo.csv (round hodo MID 1=X,9=Y; square
    // hodo MID 13=X,15=Y). Path is relative to CWD when running from DQM/.
    TButility util = TButility();
    util.LoadMapping("./mapping/mapping_test_Hodo.root");
    
    std::vector<TBcid> cids;
    std::vector<TH1F *> plots;
    for (int idx = 0; idx < channel_names.size(); idx++) {
        plots.push_back(new TH1F((TString)channel_names.at(idx), ";bin;ADC", 1000, 0, 1000));
        cids.push_back(util.GetCID(channel_names.at(idx)));
    }
        
    // Round hodoscope test setup: MID 1 (X) + MID 9 (Y).
    // Square hodoscope test setup: MID 13 (X) + MID 15 (Y).
    TBread<TBwaveform> readerWave = TBread<TBwaveform>(fRunNum, fMaxEvent, fMaxFile, false, "/Volumes/yhep/scratch/YUdaq", {1, 9, 13, 15});
    
    // Set Maximum event
    if (fMaxEvent == -1 || fMaxEvent > readerWave.GetMaxEvent())
    fMaxEvent = readerWave.GetMaxEvent();
    
    // Start event loop
    for (int i = 0; i < fMaxEvent; i++) {
        printProgress(i, fMaxEvent);
        // Load event
        TBevt<TBwaveform> aEvent = readerWave.GetAnEvent();
        // Get waveform of certain channel we want to use
        // filling plots
        for (int idx = 0; idx < plots.size(); idx++) {
            auto single_waveform = aEvent.GetData(cids.at(idx)).waveform();
            
            std::vector<float> avgTimeStruc = GetAvg(single_waveform, fMaxEvent);
            for (int bin = 1; bin < 1001; bin++) {
                plots.at(idx)->Fill(bin, avgTimeStruc.at(bin));
            }
        }
    } // end of event loop

    TCanvas *c = new TCanvas("c", "c", 1000, 800);
    c->cd();
        
    TLegend *leg = new TLegend(0.75, 0.2, 0.9, 0.4);

    // Auto-scale the shared y-range to the actual data spread instead of the
    // fixed [1000,4096] full-ADC range -- with sparse per-fiber occupancy
    // (e.g. a hodoscope fiber only fires in a fraction of events) the
    // event-averaged dip is tiny and gets flattened out at full ADC scale.
    double globalMin = 1e18, globalMax = -1e18;
    for (auto* p : plots) {
        globalMin = std::min(globalMin, p->GetMinimum());
        globalMax = std::max(globalMax, p->GetMaximum());
    }
    const double margin = 0.1 * (globalMax - globalMin);
    const double yLo = globalMin - margin;
    const double yHi = globalMax + margin;

    for (int idx = 0; idx < plots.size(); idx++) {
        plots.at(idx)->SetLineWidth(2);
        plots.at(idx)->SetLineColor(myColorPalette.at(idx));
        plots.at(idx)->GetYaxis()->SetRangeUser(yLo, yHi);
        
        c->cd();
        if (idx == 0) plots.at(idx)->Draw("Hist");
        else plots.at(idx)->Draw("Hist & sames");
        
        if (drawLegend) leg->AddEntry(plots.at(idx), channel_names.at(idx).c_str(), "l");
        c->Update();
    }
    
    // Skip the legend for large channel counts (e.g. all 122 hodoscope
    // fibers) -- it would badly overflow the fixed legend box and isn't
    // readable anyway; the overlay itself still shows the aggregate timing.
    if (drawLegend) leg->Draw("sames");
    c->Update();
    const std::string outBase = "./Avg/Run_" + std::to_string(fRunNum) + "/" + full_channel_name;
    c->SaveAs((TString)(outBase + ".png"));

    // Also write a ROOT file with the canvas + every individual per-channel
    // histogram, so it can be reopened (e.g. TBrowser, or `new TCanvas` +
    // `->Draw()`) and zoomed/panned interactively instead of being limited
    // to the flattened auto-scaled PNG.
    TFile *outRoot = new TFile((TString)(outBase + ".root"), "RECREATE");
    outRoot->cd();
    c->Write("c_overlay");
    for (int idx = 0; idx < plots.size(); idx++) plots.at(idx)->Write();
    outRoot->Close();
    std::cout << "\n[draw_Avg] Wrote " << outBase << ".root ("
              << plots.size() << " channel histograms + overlay canvas)\n";

    return 0;
}