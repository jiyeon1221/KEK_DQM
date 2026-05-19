#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>

#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <mach/mach_types.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>

#include <sys/types.h>
#include <sys/sysctl.h>

#include "TBsingleWaveform.h"

#include "GuiTypes.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TPaveStats.h"
#include "TROOT.h"


TBsingleWaveform::TBsingleWaveform(ObjectCollection* fObj) {
  // Config file path: prefer --Config <path> from the CLI; fall back to
  // the legacy hard-coded autoTB location.
  std::string config_path = "/Users/yhep/autoTB/config_general.yml";
  std::string config_arg;
  fObj->GetVariable("Config", &config_arg);
  if (!config_arg.empty() && config_arg != "null")
    config_path = config_arg;

  fConfig = TBconfig(config_path);
  const YAML::Node fConfig_YAML = fConfig.GetConfig();

  fBaseDir = fConfig_YAML["BaseDirectory"].as<std::string>();
  std::string fMapping = fConfig_YAML["Mapping"].as<std::string>();

  fUtility = TButility(fMapping);

  fObj->GetVariable("RunNumber", &fRunNum);
  fObj->GetVariable("MaxEvent", &fMaxEvent);
  fObj->GetVariable("SkipEvent", &fSkipEvent);
  
  std::vector<std::string> tempModuleVec;
  fObj->GetVector("module", &tempModuleVec);
  
  // Expand module names: "T1" -> ["T1-C", "T1-S"], "T1-C" -> ["T1-C"]
  for (const auto& moduleName : tempModuleVec) {
    // Check if already has -C or -S suffix
    if (moduleName.find("-C") != std::string::npos || moduleName.find("-S") != std::string::npos) {
      // Already specified, use as is
      fNametoPlot.push_back(moduleName);
    } else {
      // Tower name only (e.g., "T1"), expand to both C and S
      fNametoPlot.push_back(moduleName + "-C");
      fNametoPlot.push_back(moduleName + "-S");
    }
  }

  if (fSkipEvent == -1) fSkipEvent = 0;

  // Create waveforms directory if it doesn't exist
  gSystem->mkdir("./output/waveforms", kTRUE);
  
  fOutputName = (TString)Form("./output/waveforms/Run%d_SingleWaveform", fRunNum);

  init();
}

void TBsingleWaveform::init() {

  // Enable batch mode for ROOT
  gROOT->SetBatch(kTRUE);

  std::vector<int> fColorVec = {
    TColor::GetColor("#5790fc"),
    TColor::GetColor("#f89c20"),
    TColor::GetColor("#e42536"),
    TColor::GetColor("#964a8b"),
    TColor::GetColor("#9c9ca1"),
    TColor::GetColor("#7a21dd")
  };

  fCanvas = new TCanvas("canvas", "canvas", 800, 800);

  fMainFrame = new TH1D("main", ";Bin;ADC", 1000, 0.5, 1000.5);
  fMainFrame->SetStats(0);

  fLeg = new TLegend(0.7, 0.2, 0.9, 0.5);
  fLeg->SetFillStyle(0);
  fLeg->SetBorderSize(0);
  fLeg->SetTextFont(42);

  for (int i = 0; i < fNametoPlot.size(); i++) {

    std::string aName = fNametoPlot.at(i);
    TBcid aCID = fUtility.GetCID(aName);
    fCIDtoPlot.push_back(aCID);

    fHistWaveform.push_back(new TH1D((TString)Form("%s_waveform", aName.c_str()), ";Bin;ADC", 1000, 0.5, 1000.5));
    fHistWaveform.at(i)->SetLineColor(fColorVec.at(i));
    fHistWaveform.at(i)->SetLineWidth(2);
    fHistWaveform.at(i)->SetStats(0);

    fLeg->AddEntry(fHistWaveform.at(i), aName.c_str(), "l");
  }
}

void TBsingleWaveform::GetFormattedRamInfo() {

    // Total physical memory
    int64_t physical_memory;
    size_t length = sizeof(physical_memory);
    sysctlbyname("hw.memsize", &physical_memory, &length, NULL, 0);
    double total_memory_GB = static_cast<double>(physical_memory) / (1024 * 1024 * 1024);

    // Memory usage by this process
    task_basic_info_data_t info;
    mach_msg_type_number_t info_count = TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&info, &info_count) == KERN_SUCCESS) {
        double process_memory_GB = static_cast<double>(info.resident_size) / (1024 * 1024 * 1024);

        // system memory usage
        vm_size_t page_size;
        mach_port_t mach_port = mach_host_self();
        vm_statistics64_data_t vm_stats;
        mach_msg_type_number_t count = sizeof(vm_stats) / sizeof(natural_t);
        if (host_page_size(mach_port, &page_size) == KERN_SUCCESS &&
            host_statistics64(mach_port, HOST_VM_INFO, (host_info64_t)&vm_stats, &count) == KERN_SUCCESS) {
            double free_memory_GB = static_cast<double>(vm_stats.free_count * page_size) / (1024 * 1024 * 1024);
            double used_memory_GB = total_memory_GB - free_memory_GB;


            printf("%.1f GB / %.1f GB (%0.2f %%) | Current Process: %.2f MB (%.2f %%)",
              used_memory_GB, total_memory_GB, (used_memory_GB / total_memory_GB * 100),
              process_memory_GB * 1024., (process_memory_GB / total_memory_GB * 100));
        }
    }
}

void TBsingleWaveform::Loop() {
  
  ANSI_CODE ANSI = ANSI_CODE();

  std::vector<std::string> fOutputFiles;
  std::vector<int> tUniqueMID = GetUniqueMID();
  TBread<TBwaveform> readerWave = TBread<TBwaveform>(fRunNum, fSkipEvent + fMaxEvent, 1, false, fBaseDir, tUniqueMID);

  if (fMaxEvent > readerWave.GetMaxEvent()) fMaxEvent = readerWave.GetMaxEvent();
  if (fMaxEvent - fSkipEvent > 100) fMaxEvent = fSkipEvent + 100;
  if (fMaxEvent - fSkipEvent < 0) fMaxEvent = fSkipEvent + 100;

  std::chrono::time_point time_begin = std::chrono::system_clock::now();
  for (int i = 0; i < fMaxEvent; i++) {

    TBevt<TBwaveform> anEvent = readerWave.GetAnEvent();

    if (i < fSkipEvent) continue;

    if (i > fSkipEvent && i % 10 == 0) {

      std::chrono::duration time_taken = std::chrono::system_clock::now() - time_begin; // delete
      float percent_done = 1. * (float)(i - fSkipEvent) / (float)(fMaxEvent - fSkipEvent);
      std::chrono::duration time_left = time_taken * (1 / percent_done - 1);
      std::chrono::minutes minutes_left = std::chrono::duration_cast<std::chrono::minutes>(time_left);
      std::chrono::seconds seconds_left = std::chrono::duration_cast<std::chrono::seconds>(time_left - minutes_left);
      std::cout << "\r\033[F" //+ ANSI.HIGHLIGHTED_GREEN + ANSI.BLACK
                << " " << i << " / " << fMaxEvent << " events  " << minutes_left.count() << ":";
      printf("%02d left (%.1f %%) | ", int(seconds_left.count()), percent_done * 100);
      GetFormattedRamInfo();

      std::cout << ANSI.END << std::endl;
    }

    for (int iCh = 0; iCh < fHistWaveform.size(); iCh++) {

      auto aWaveform = anEvent.GetData(fCIDtoPlot.at(iCh)).waveform();
      for (int iBin = 1; iBin <= 1000; iBin++) 
        fHistWaveform.at(iCh)->SetBinContent(iBin, aWaveform.at(iBin));
  
    }

    SetMaximum();
    
    int currentEvent = readerWave.GetCurrentEvent();
    fMainFrame->SetTitle(Form("Run %d, Event %d", fRunNum, currentEvent));

    fCanvas->cd();
    fMainFrame->Draw();
    for (int iCh = 0; iCh < fHistWaveform.size(); iCh++) {
      fHistWaveform.at(iCh)->Draw("Hist & sames");
    }
    fLeg->Draw();
    fCanvas->Update();
    
    // Save each event as individual PNG file
    TString pngFileName = Form("%s_evt%d.png", fOutputName.Data(), currentEvent);
    fCanvas->SaveAs(pngFileName);
    fOutputFiles.push_back(pngFileName.Data());

    for (int iCh = 0; iCh < fHistWaveform.size(); iCh++)
      fHistWaveform.at(iCh)->Reset("ICES");
  }

  // Save file list as JSON for web UI in output root directory
  TString jsonFileName = Form("./output/Run%d_SingleWaveform.json", fRunNum);
  std::ofstream jsonFile(jsonFileName.Data());
  jsonFile << "{\"files\":[";
  for (size_t i = 0; i < fOutputFiles.size(); i++) {
    // Store relative paths from output directory
    std::string relativePath = fOutputFiles[i];
    size_t pos = relativePath.find("./output/");
    if (pos != std::string::npos) {
      relativePath = relativePath.substr(pos + 9); // Remove "./output/" prefix
    }
    jsonFile << "\"" << relativePath << "\"";
    if (i < fOutputFiles.size() - 1) jsonFile << ",";
  }
  jsonFile << "],\"total\":" << fOutputFiles.size() << "}";
  jsonFile.close();
  
  std::cout << "\n✅ Generated " << fOutputFiles.size() << " PNG files in ./output/waveforms/" << std::endl;
  std::cout << "📄 File list saved to " << jsonFileName.Data() << std::endl;
}

void TBsingleWaveform::SetMaximum() {

  float fMax = -9999;
  float fMin = 9999;
  
  for (int iCh = 0; iCh < fHistWaveform.size(); iCh++) {
    if (fHistWaveform.at(iCh)->GetMaximum() > fMax) fMax = fHistWaveform.at(iCh)->GetMaximum();
    if (fHistWaveform.at(iCh)->GetMinimum() < fMin) fMin = fHistWaveform.at(iCh)->GetMinimum();
  }

  float fWindow = fMax - fMin;
  fMax = fMax + fWindow * 0.1;
  fMin = fMin - fWindow * 0.1;

  fMainFrame->GetYaxis()->SetRangeUser(fMin, fMax);
}

std::vector<int> TBsingleWaveform::GetUniqueMID() {

  return fUtility.GetUniqueMID(fCIDtoPlot);
}
