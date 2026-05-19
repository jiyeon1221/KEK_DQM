#include "TBplotengine.h"
#include "GuiTypes.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TPaveStats.h"
#include "TBufferJSON.h"

#include <fstream>
#include <cstdio>
#include <climits>

TBplotengine::TBplotengine(const YAML::Node fConfig_, int fRunNum_, bool fLive_, bool fDraw_, TButility fUtility_)
: fConfig(fConfig_), fRunNum(fRunNum_), fLive(fLive_), fDraw(fDraw_), fUtility(fUtility_), fCaseName(""), fAuxCut(false)
{}

void TBplotengine::init() {

  fIsFirst = true;
  fUsingAUX = false;
  gStyle->SetPalette(kRainBow);

  std::vector<int> fColorVec = {
    TColor::GetColor("#5790fc"),
    TColor::GetColor("#f89c20"),
    TColor::GetColor("#009E73"),
    TColor::GetColor("#e42536"),
    TColor::GetColor("#F0E442"),
    TColor::GetColor("#7a21dd"),
    TColor::GetColor("#964a8b"),
    TColor::GetColor("#661100"),
    TColor::GetColor("#9c9ca1"),
    TColor::GetColor("#44AA99"),
  };

  if (fCaseName == "single") {

    if (fCalcInfo == TBplotengine::CalcInfo::kAvgTimeStruc) {
      fLeg = new TLegend(0.7, 0.2, 0.9, 0.5);
     	fLeg->SetFillStyle(0);
     	fLeg->SetBorderSize(0);
     	fLeg->SetTextFont(42);
    }


    for (int i = 0; i < fNametoPlot.size(); i++) {
      // std::string aName = fUtility.GetName(aCID);
      std::string aName = fNametoPlot.at(i);
      TBcid aCID = fUtility.GetCID(aName);
      fCIDtoPlot_Ceren.push_back(aCID);

      TButility::mod_info aInfo = fUtility.GetInfo(aCID);
      // std::cout << aName << " "; aCID.print();

      if (fCalcInfo == TBplotengine::CalcInfo::kIntADC || fCalcInfo == TBplotengine::CalcInfo::kPeakADC) {
        std::vector<int> interval = fConfig[aName].as<std::vector<int>>();
        fPlotter_Ceren.push_back(TBplotengine::PlotInfo(aCID, aName, aInfo, interval.at(0), interval.at(1)));

        if (fCalcInfo == TBplotengine::CalcInfo::kIntADC) {
         if (aName.find("LC") != std::string::npos) fPlotter_Ceren.at(i).SetPlot(new TH1D((TString)(aName), ";IntADC;nEvents", 1200, -10000., 50000.));
         else fPlotter_Ceren.at(i).SetPlot(new TH1D((TString)(aName), ";IntADC;nEvents", 440, -30000., 300000.));
        }

        if (fCalcInfo == TBplotengine::CalcInfo::kPeakADC)
          fPlotter_Ceren.at(i).SetPlot(new TH1D((TString)(aName), ";PeakADC;nEvents", 1152, -512., 4096.));

        fPlotter_Ceren.at(i).hist1D->SetLineColor(fColorVec.at(i));
        fPlotter_Ceren.at(i).hist1D->SetLineWidth(2);

      } else if (fCalcInfo == TBplotengine::CalcInfo::kAvgTimeStruc) {
        fPlotter_Ceren.push_back(TBplotengine::PlotInfo(aCID, aName, aInfo, 0, 0));
        fPlotter_Ceren.at(i).SetPlot(new TH1D((TString)(aName), ";Bin;ADC", 1000, 0.5, 1000.5));
        fPlotter_Ceren.at(i).hist1D->SetLineColor(fColorVec.at(i));
        fPlotter_Ceren.at(i).hist1D->SetLineWidth(2);
        fPlotter_Ceren.at(i).hist1D->SetStats(0);

      } else if (fCalcInfo == TBplotengine::CalcInfo::kOverlay) {
        fPlotter_Ceren.push_back(TBplotengine::PlotInfo(aCID, aName, aInfo, 0, 0));
        fPlotter_Ceren.at(i).SetPlot(new TH2D((TString)(aName), (TString)"Run " + std::to_string(fRunNum) + ";Bin;ADC", 1024, 0., 1024., 4096, 0., 4096.));
        fPlotter_Ceren.at(i).hist2D->SetStats(0);

      } else {
        fPlotter_Ceren.push_back(TBplotengine::PlotInfo(aCID, aName, aInfo));
      }
    }

    if (fCalcInfo == TBplotengine::CalcInfo::kAvgTimeStruc) {
      fMainFrame = new TH1D("frame", (TString)"Run " + std::to_string(fRunNum) + ";Bin;ADC", 1000, 0.5, 1000.5);
      fMainFrame->SetStats(0);
    }

    if (fCalcInfo == TBplotengine::CalcInfo::kIntADC) {
      fMainFrame = new TH1D("frame", (TString)"Run " + std::to_string(fRunNum) + ";IntADC;nEvents", 440, -30000., 300000.);
      fMainFrame->SetStats(0);
    }

    if (fCalcInfo == TBplotengine::CalcInfo::kPeakADC) {
      fMainFrame = new TH1D("frame", (TString)"Run " + std::to_string(fRunNum) + ";PeakADC;nEvents", 288, -512., 4096.);
      fMainFrame->SetStats(0);
    }

    fCanvas = new TCanvas("fCanvasPlot", "fCanvasPlot", 1400, 1400);

    Draw();
  } else if (fCaseName == "full") {

    fCanvasFull.push_back(new TCanvas("fCanvasHeatmap", "fCanvasHeatmap", 2700, 1000));
    fCanvasFull.at(0)->Divide(2, 1);

    auto tPadLeft = fCanvasFull.at(0)->cd(1);
    tPadLeft->SetRightMargin(0.13);

    auto tPadRight = fCanvasFull.at(0)->cd(2);
    tPadRight->SetRightMargin(0.13);

    // Per-tower canvases are allocated inside init_Generic() after the tower
    // list is discovered from the mapping (so the count matches the actual
    // mapping geometry: 9 for KEK2026 T1..T9, 36 for TB2025 M1-T1..M9-T4, etc.).
    init_Generic();
  } else if (fCaseName == "module") {

    // One canvas with the tower-grid layout (N_x × N_y from the mapping),
    // every pad holding a tower's IntADC/PeakADC distribution (C blue + S red
    // overlaid). The canvas and pad geometry are constructed inside
    // init_module() because the grid dimensions are mapping-driven.
    init_module();
  } else if (fCaseName == "heatmap") {

    // For now only MCPPMT (C1..C64 + S1..S64) is wired up. SiPM and other
    // detectors will follow once their channel naming is finalized.
    if (fModule != "MCPPMT") {
      std::cerr << "[TBplotengine] --type heatmap currently supports only "
                << "--module MCPPMT (got --module '" << fModule << "'). "
                << "Skipping init." << std::endl;
      return;
    }

    fCanvasFull.push_back(new TCanvas("fCanvasMCPPMT", "fCanvasMCPPMT", 2700, 1000));
    fCanvasFull.at(0)->Divide(2, 1);

    auto tPadLeft = fCanvasFull.at(0)->cd(1);
    tPadLeft->SetRightMargin(0.13);

    auto tPadRight = fCanvasFull.at(0)->cd(2);
    tPadRight->SetRightMargin(0.13);

    init_MCPPMT();
  }
}


void TBplotengine::init_Generic() {

  // ── Data-driven tower discovery ───────────────────────────────────────────
  // The previous implementation hardcoded plotVec = {"T1",..,"T9"} and a 3x3
  // grid, which only matched the KEK2026 layout. We now enumerate the loaded
  // mapping (TButility::GetNameInfo()) and pick up anything that looks like a
  // tower channel: names ending in "-C" (Cherenkov side) or "-S" (Scintillator
  // side) with valid (row, col). This covers, with a single code path:
  //   * KEK2026 KEK_DQM mapping: T1-C..T9-S        -> 3x3 grid, 9 towers
  //   * TB2025 legacy mapping:   M1-T1-C..M9-T4-S  -> 6x6 grid, 36 towers
  //   * any future "<prefix>-<tower>-{C,S}" naming
  // The hyphen-suffix filter intentionally excludes MCPPMT C1..C64 / S1..S64
  // (which have no "-C"/"-S" suffix), so a mapping that contains both MCPPMT
  // and tower entries will still produce the correct tower-only full view.
  // (row, col) for the 2D heatmap come from the mapping CSV; the grid size is
  // computed from max(row), max(col) so the histogram auto-sizes to whatever
  // geometry the mapping describes.
  std::vector<std::string> cerenNames, scintNames;
  int maxRow = 0, maxCol = 0;

  for (const auto& kv : fUtility.GetNameInfo()) {
    const std::string name = std::string(kv.first.Data());
    const auto& info = kv.second;
    if (info.row <= 0 || info.col <= 0) continue;
    if (name.size() < 2) continue;
    const std::string suffix = name.substr(name.size() - 2);
    if (suffix == "-C" && info.isCeren == 1) {
      cerenNames.push_back(name);
      maxRow = std::max(maxRow, info.row);
      maxCol = std::max(maxCol, info.col);
    } else if (suffix == "-S" && info.isCeren == 0) {
      scintNames.push_back(name);
      maxRow = std::max(maxRow, info.row);
      maxCol = std::max(maxCol, info.col);
    }
  }

  // Lexicographic order so per-tower canvas indices are deterministic across
  // runs regardless of the mapping's insertion order, and so Ceren[i] / Scint[i]
  // line up by name root (T1-C with T1-S, M1-T1-C with M1-T1-S, ...).
  std::sort(cerenNames.begin(), cerenNames.end());
  std::sort(scintNames.begin(), scintNames.end());

  if (cerenNames.empty() && scintNames.empty()) {
    std::cerr << "[TBplotengine] init_Generic: no tower-like channels found in "
              << "mapping (looked for names ending in '-C' or '-S' with valid "
              << "row/col). The CERENKOV/SCINTILLATION heatmaps will be empty."
              << std::endl;
  }
  if (cerenNames.size() != scintNames.size()) {
    std::cerr << "[TBplotengine] init_Generic: Cherenkov and Scintillator "
              << "tower counts differ in the mapping ("
              << cerenNames.size() << " vs " << scintNames.size() << "). "
              << "Per-tower 1D canvases will use min(C, S)." << std::endl;
  }

  for (size_t i = 0; i < cerenNames.size(); ++i) {
    const std::string& aCName = cerenNames[i];
    TBcid aCCID = fUtility.GetCID(aCName);
    TButility::mod_info aCInfo = fUtility.GetInfo(aCName);

    fCIDtoPlot_Ceren.push_back(aCCID);

    std::vector<int> intervalC = fConfig[aCName].as<std::vector<int>>();
    fPlotter_Ceren.push_back(TBplotengine::PlotInfo(aCCID, aCName, aCInfo, intervalC.at(0), intervalC.at(1)));

    if (fCalcInfo == TBplotengine::CalcInfo::kIntADC)
      fPlotter_Ceren.at(i).SetPlot(new TH1D((TString)(aCName), ";IntADC;nEvents", 440, -30000., 300000.));

    if (fCalcInfo == TBplotengine::CalcInfo::kPeakADC)
      fPlotter_Ceren.at(i).SetPlot(new TH1D((TString)(aCName), ";PeakADC;nEvents", 288, -512., 4096.));

    fPlotter_Ceren.at(i).hist1D->SetLineWidth(2);
    fPlotter_Ceren.at(i).hist1D->SetLineColor(kBlue);
  }

  for (size_t i = 0; i < scintNames.size(); ++i) {
    const std::string& aSName = scintNames[i];
    TBcid aSCID = fUtility.GetCID(aSName);
    TButility::mod_info aSInfo = fUtility.GetInfo(aSName);

    fCIDtoPlot_Scint.push_back(aSCID);

    std::vector<int> intervalS = fConfig[aSName].as<std::vector<int>>();
    fPlotter_Scint.push_back(TBplotengine::PlotInfo(aSCID, aSName, aSInfo, intervalS.at(0), intervalS.at(1)));

    if (fCalcInfo == TBplotengine::CalcInfo::kIntADC)
      fPlotter_Scint.at(i).SetPlot(new TH1D((TString)(aSName), ";IntADC;nEvents", 440, -30000., 300000.));

    if (fCalcInfo == TBplotengine::CalcInfo::kPeakADC)
      fPlotter_Scint.at(i).SetPlot(new TH1D((TString)(aSName), ";PeakADC;nEvents", 288, -512., 4096.));

    fPlotter_Scint.at(i).hist1D->SetLineWidth(2);
    fPlotter_Scint.at(i).hist1D->SetLineColor(kRed);
  }

  // Grid size derived from the mapping. Both axes use the same span so the
  // heatmap is square-ish; for asymmetric layouts the unused cells just stay
  // empty (their bin labels are still drawn).
  const int gridX = std::max(1, maxRow);
  const int gridY = std::max(1, maxCol);

  f2DHistCeren = new TH2D("CERENKOV", "CERENKOV;;", gridX, 0.5, gridX + 0.5, gridY, 0.5, gridY + 0.5);
  f2DHistCeren->SetStats(0);

  f2DHistScint = new TH2D("SCINTILLATION", "SCINTILLATION;;", gridX, 0.5, gridX + 0.5, gridY, 0.5, gridY + 0.5);
  f2DHistScint->SetStats(0);

  for (int i = 1; i <= gridX; ++i) {
    f2DHistCeren->GetXaxis()->SetBinLabel(i, std::to_string(i).c_str());
    f2DHistScint->GetXaxis()->SetBinLabel(i, std::to_string(i).c_str());
  }
  for (int i = 1; i <= gridY; ++i) {
    f2DHistCeren->GetYaxis()->SetBinLabel(i, std::to_string(i).c_str());
    f2DHistScint->GetYaxis()->SetBinLabel(i, std::to_string(i).c_str());
  }

  // Per-tower 1D canvases, sized to the smaller of the two lists so Draw/Update
  // can safely pair Ceren[idx] with Scint[idx] without out-of-range access.
  const size_t nTowers = std::min(fPlotter_Ceren.size(), fPlotter_Scint.size());
  for (size_t i = 0; i < nTowers; ++i) {
    std::string aCanvasName = "fCanvas_Tower" + std::to_string(i + 1);
    fCanvasFull.push_back(new TCanvas((TString)aCanvasName, (TString)aCanvasName, 1400, 700));
    fCanvasFull.back()->Divide(2, 1);
  }

  Draw();
}


void TBplotengine::init_module() {

  // ── --type module: per-tower distribution grid ────────────────────────────
  // This mirrors init_Generic's tower discovery (mapping-driven, name suffix
  // -C/-S, valid (row,col)) but with two differences:
  //   1. No 2D heatmap is built. Each pad in the canvas holds a 1D IntADC /
  //      PeakADC distribution instead of a colored bin.
  //   2. An optional --module <prefix> filter restricts the towers to those
  //      whose names start with "<prefix>-". This makes one binary serve:
  //        * KEK 2026: fModule == ""        → all 9 towers T1..T9 → 3x3 grid
  //        * CERN  2025: fModule == "M1"    → M1's 4 towers       → 2x2 grid
  //      The grid is auto-sized from min..max of (row, col) within the
  //      filtered set, so M1 (at rows 1..2, cols 5..6 of the 6x6 master grid)
  //      collapses to a clean 2x2 view local to that module.
  // The pad index is computed from (info.row, info.col) using ROOT's
  // Divide(nx, ny) numbering (left-to-right, top-to-bottom from pad 1), so
  // physical tower positions on screen match the heatmap exactly.

  const bool hasFilter = !fModule.empty();
  const std::string filterPrefix = hasFilter ? fModule + "-" : "";

  std::vector<std::string> cerenNames, scintNames;
  int minRow = INT_MAX, minCol = INT_MAX;
  int maxRow = 0, maxCol = 0;

  for (const auto& kv : fUtility.GetNameInfo()) {
    const std::string name = std::string(kv.first.Data());
    const auto& info = kv.second;
    if (info.row <= 0 || info.col <= 0) continue;
    if (name.size() < 2) continue;
    if (hasFilter && name.compare(0, filterPrefix.size(), filterPrefix) != 0) continue;
    const std::string suffix = name.substr(name.size() - 2);
    if (suffix == "-C" && info.isCeren == 1) {
      cerenNames.push_back(name);
    } else if (suffix == "-S" && info.isCeren == 0) {
      scintNames.push_back(name);
    } else {
      continue;
    }
    minRow = std::min(minRow, info.row);
    minCol = std::min(minCol, info.col);
    maxRow = std::max(maxRow, info.row);
    maxCol = std::max(maxCol, info.col);
  }

  std::sort(cerenNames.begin(), cerenNames.end());
  std::sort(scintNames.begin(), scintNames.end());

  if (cerenNames.empty() && scintNames.empty()) {
    std::cerr << "[TBplotengine] init_module: no tower channels found"
              << (hasFilter ? (" for prefix '" + fModule + "-'.") : ".")
              << " The module-view canvas will be empty." << std::endl;
    fGridX_module = 1;
    fGridY_module = 1;
    fCanvas = new TCanvas("fCanvasModule", "fCanvasModule", 1400, 1400);
    Draw();
    return;
  }

  // Local grid: collapse the absolute (row, col) into 1..N starting at
  // (minRow, minCol). For the KEK case this is a no-op (min == 1 already);
  // for "--module M1" on TB2025 it produces a clean 2x2 instead of an offset
  // window into the master 6x6.
  fGridX_module = maxRow - minRow + 1;
  fGridY_module = maxCol - minCol + 1;
  const int rowOffset = minRow - 1;
  const int colOffset = minCol - 1;

  // Lay out the canvas. The pad numbering goes left-to-right top-to-bottom
  // starting at 1, so Divide(nx, ny) maps to (cols=nx, rows=ny). We translate
  // mapping coordinates → pad index in Draw/Update via:
  //    pad = (gridY - localCol) * gridX + localRow
  fCanvas = new TCanvas("fCanvasModule", "fCanvasModule",
                        std::max(700, 400 * fGridX_module),
                        std::max(700, 400 * fGridY_module));
  fCanvas->Divide(fGridX_module, fGridY_module);

  for (size_t i = 0; i < cerenNames.size(); ++i) {
    const std::string& aCName = cerenNames[i];
    TBcid aCCID = fUtility.GetCID(aCName);
    TButility::mod_info aCInfo = fUtility.GetInfo(aCName);
    // Bake the local offset back into info so Draw/Update can use it directly.
    aCInfo.row -= rowOffset;
    aCInfo.col -= colOffset;

    fCIDtoPlot_Ceren.push_back(aCCID);

    std::vector<int> intervalC = fConfig[aCName].as<std::vector<int>>();
    fPlotter_Ceren.push_back(TBplotengine::PlotInfo(aCCID, aCName, aCInfo, intervalC.at(0), intervalC.at(1)));

    if (fCalcInfo == TBplotengine::CalcInfo::kIntADC)
      fPlotter_Ceren.back().SetPlot(new TH1D((TString)(aCName), (TString)(aCName + ";IntADC;nEvents"), 440, -30000., 300000.));

    if (fCalcInfo == TBplotengine::CalcInfo::kPeakADC)
      fPlotter_Ceren.back().SetPlot(new TH1D((TString)(aCName), (TString)(aCName + ";PeakADC;nEvents"), 288, -512., 4096.));

    fPlotter_Ceren.back().hist1D->SetLineWidth(2);
    fPlotter_Ceren.back().hist1D->SetLineColor(kBlue);
  }

  for (size_t i = 0; i < scintNames.size(); ++i) {
    const std::string& aSName = scintNames[i];
    TBcid aSCID = fUtility.GetCID(aSName);
    TButility::mod_info aSInfo = fUtility.GetInfo(aSName);
    aSInfo.row -= rowOffset;
    aSInfo.col -= colOffset;

    fCIDtoPlot_Scint.push_back(aSCID);

    std::vector<int> intervalS = fConfig[aSName].as<std::vector<int>>();
    fPlotter_Scint.push_back(TBplotengine::PlotInfo(aSCID, aSName, aSInfo, intervalS.at(0), intervalS.at(1)));

    if (fCalcInfo == TBplotengine::CalcInfo::kIntADC)
      fPlotter_Scint.back().SetPlot(new TH1D((TString)(aSName), (TString)(aSName + ";IntADC;nEvents"), 440, -30000., 300000.));

    if (fCalcInfo == TBplotengine::CalcInfo::kPeakADC)
      fPlotter_Scint.back().SetPlot(new TH1D((TString)(aSName), (TString)(aSName + ";PeakADC;nEvents"), 288, -512., 4096.));

    fPlotter_Scint.back().hist1D->SetLineWidth(2);
    fPlotter_Scint.back().hist1D->SetLineColor(kRed);
  }

  Draw();
}


void TBplotengine::init_MCPPMT() {

  // Channel-name list is hardcoded (C1..C64 / S1..S64). The MCPPMT layout is
  // not expected to grow new C*/S* names; if it ever does, this is the only
  // place to extend.
  for (int i = 1; i <= 64; ++i) {
    const std::string aCName = "C" + std::to_string(i);
    const TBcid aCCID = fUtility.GetCID(aCName);
    const TButility::mod_info aCInfo = fUtility.GetInfo(aCName);

    fCIDtoPlot_Ceren.push_back(aCCID);

    const std::vector<int> intervalC = fConfig[aCName].as<std::vector<int>>();
    fPlotter_Ceren.push_back(TBplotengine::PlotInfo(aCCID, aCName, aCInfo, intervalC.at(0), intervalC.at(1)));

    if (fCalcInfo == TBplotengine::CalcInfo::kIntADC)
      fPlotter_Ceren.back().SetPlot(new TH1D((TString)aCName, ";IntADC;nEvents", 440, -30000., 300000.));

    if (fCalcInfo == TBplotengine::CalcInfo::kPeakADC)
      fPlotter_Ceren.back().SetPlot(new TH1D((TString)aCName, ";PeakADC;nEvents", 288, -512., 4096.));

    fPlotter_Ceren.back().hist1D->SetLineWidth(2);
    fPlotter_Ceren.back().hist1D->SetLineColor(kBlue);

    const std::string aSName = "S" + std::to_string(i);
    const TBcid aSCID = fUtility.GetCID(aSName);
    const TButility::mod_info aSInfo = fUtility.GetInfo(aSName);

    fCIDtoPlot_Scint.push_back(aSCID);

    const std::vector<int> intervalS = fConfig[aSName].as<std::vector<int>>();
    fPlotter_Scint.push_back(TBplotengine::PlotInfo(aSCID, aSName, aSInfo, intervalS.at(0), intervalS.at(1)));

    if (fCalcInfo == TBplotengine::CalcInfo::kIntADC)
      fPlotter_Scint.back().SetPlot(new TH1D((TString)aSName, ";IntADC;nEvents", 440, -30000., 300000.));

    if (fCalcInfo == TBplotengine::CalcInfo::kPeakADC)
      fPlotter_Scint.back().SetPlot(new TH1D((TString)aSName, ";PeakADC;nEvents", 288, -512., 4096.));

    fPlotter_Scint.back().hist1D->SetLineWidth(2);
    fPlotter_Scint.back().hist1D->SetLineColor(kRed);
  }

  // 8x8 heatmaps. Bin coords use (column, row) to match the (col, row) order
  // already established by the full-mode Fill at line ~260.
  f2DHistCeren = new TH2D("MCPPMT_C", "MCPPMT C;column;row", 8, 0.5, 8.5, 8, 0.5, 8.5);
  f2DHistCeren->SetStats(0);

  f2DHistScint = new TH2D("MCPPMT_S", "MCPPMT S;column;row", 8, 0.5, 8.5, 8, 0.5, 8.5);
  f2DHistScint->SetStats(0);

  for (int i = 1; i <= 8; ++i) {
    f2DHistCeren->GetXaxis()->SetBinLabel(i, std::to_string(i).c_str());
    f2DHistCeren->GetYaxis()->SetBinLabel(i, std::to_string(i).c_str());
    f2DHistScint->GetXaxis()->SetBinLabel(i, std::to_string(i).c_str());
    f2DHistScint->GetYaxis()->SetBinLabel(i, std::to_string(i).c_str());
  }

  Draw();
}


double TBplotengine::GetPeakADC(std::vector<short> waveform, int xInit, int xFin) {
  double ped = 0;
  for (int i = 1; i < 101; i++)
    ped += (double)waveform.at(i) / 100.;

  std::vector<double> pedCorWave;
  for (int i = xInit; i < xFin; i++)
    pedCorWave.push_back(ped - (double)waveform.at(i));

  return *std::max_element(pedCorWave.begin(), pedCorWave.end());
}

double TBplotengine::GetIntADC(std::vector<short> waveform, int xInit, int xFin) {

  double ped = 0;
  for (int i = 1; i < 101; i++)
    ped += (double)waveform.at(i) / 100.;

  double intADC_ = 0;
  for (int i = xInit; i < xFin; i++)
    intADC_ += ped - (double)waveform.at(i);

  return intADC_;
}

void TBplotengine::PrintInfo() {

}

void TBplotengine::Fill(TBevt<TBwaveform> anEvent) {

  if (fCaseName == "single") {
    if (fCalcInfo == TBplotengine::CalcInfo::kIntADC || fCalcInfo == TBplotengine::CalcInfo::kPeakADC) {
      for (int i = 0; i < fPlotter_Ceren.size(); i++) {
        double value = GetValue(anEvent.GetData(fPlotter_Ceren.at(i).cid).waveform(), fPlotter_Ceren.at(i).xInit, fPlotter_Ceren.at(i).xFin);
        fPlotter_Ceren.at(i).hist1D->Fill(value);
      }
    } else if (fCalcInfo == TBplotengine::CalcInfo::kAvgTimeStruc) {
      for (int i = 0; i < fPlotter_Ceren.size(); i++) {
        auto tWave = anEvent.GetData(fPlotter_Ceren.at(i).cid).waveform();
        for (int j = 1; j <= 1000; j++) {
          fPlotter_Ceren.at(i).hist1D->Fill(j, tWave.at(j));
        }
        fPlotter_Ceren.at(i).xInit++;
      }
    } else if (fCalcInfo == TBplotengine::CalcInfo::kOverlay) {
      for (int i = 0; i < fPlotter_Ceren.size(); i++) {
        auto tWave = anEvent.GetData(fPlotter_Ceren.at(i).cid).waveform();
        for (int j = 0; j < tWave.size(); j++) {
          fPlotter_Ceren.at(i).hist2D->Fill(j, tWave.at(j));
        }
      }
    }

  } else if (fCaseName == "full" || fCaseName == "heatmap") {
    // heatmap (MCPPMT) reuses the full-mode Fill: each channel gets a 1D
    // distribution accumulator plus a running fill into the 2D heatmap keyed
    // by (col, row) from the mapping CSV. Update() then overwrites the 2D
    // bins with the per-channel hist1D->GetMean() value.

    for (int i = 0; i < fPlotter_Ceren.size(); i++) {
      double value = GetValue(anEvent.GetData(fPlotter_Ceren.at(i).cid).waveform(), fPlotter_Ceren.at(i).xInit, fPlotter_Ceren.at(i).xFin);
      fPlotter_Ceren.at(i).hist1D->Fill(value);

      if (fPlotter_Ceren.at(i).info.row > 0 && fPlotter_Ceren.at(i).info.col > 0) {
        f2DHistCeren->Fill(fPlotter_Ceren.at(i).info.col, fPlotter_Ceren.at(i).info.row, value);
      }
    }

    for (int i = 0; i < fPlotter_Scint.size(); i++) {
      double value = GetValue(anEvent.GetData(fPlotter_Scint.at(i).cid).waveform(), fPlotter_Scint.at(i).xInit, fPlotter_Scint.at(i).xFin);
      fPlotter_Scint.at(i).hist1D->Fill(value);

      if (fPlotter_Scint.at(i).info.row > 0 && fPlotter_Scint.at(i).info.col > 0) {
        f2DHistScint->Fill(fPlotter_Scint.at(i).info.col, fPlotter_Scint.at(i).info.row, value);
      }
    }
  } else if (fCaseName == "module") {
    // Per-tower 1D distributions only — no 2D heatmap is built in module mode.
    for (size_t i = 0; i < fPlotter_Ceren.size(); ++i) {
      const double value = GetValue(anEvent.GetData(fPlotter_Ceren.at(i).cid).waveform(),
                                    fPlotter_Ceren.at(i).xInit, fPlotter_Ceren.at(i).xFin);
      fPlotter_Ceren.at(i).hist1D->Fill(value);
    }
    for (size_t i = 0; i < fPlotter_Scint.size(); ++i) {
      const double value = GetValue(anEvent.GetData(fPlotter_Scint.at(i).cid).waveform(),
                                    fPlotter_Scint.at(i).xInit, fPlotter_Scint.at(i).xFin);
      fPlotter_Scint.at(i).hist1D->Fill(value);
    }
  }
}

void TBplotengine::Draw() {

  if (fCalcInfo == TBplotengine::CalcInfo::kAvgTimeStruc) {
    for (int i = 0; i < fPlotter_Ceren.size(); i++)
      fLeg->AddEntry(fPlotter_Ceren.at(i).hist1D, fPlotter_Ceren.at(i).name.c_str(), "l");

  }

  if (fCaseName == "single") {
    fCanvas->cd();

    if (fCalcInfo == TBplotengine::CalcInfo::kOverlay) {
      fPlotter_Ceren.at(0).hist2D->Draw("colz");

    } else {
      fMainFrame->Draw();
      for (int i = 0; i < fPlotter_Ceren.size(); i++)
        fPlotter_Ceren.at(i).hist1D->Draw("sames");

      if (fCalcInfo == TBplotengine::CalcInfo::kAvgTimeStruc)
        fLeg->Draw("same");
    }

  } else if (fCaseName == "full" || fCaseName == "heatmap") {

    fCanvasFull.at(0)->cd(1);
    f2DHistCeren->Draw("colz text");

    fCanvasFull.at(0)->cd(2);
    f2DHistScint->Draw("colz text");

    // full mode also draws per-tower 1D distributions on dedicated canvases.
    // heatmap mode only produces the 2D heatmap pair, so skip the per-channel
    // canvas loop (we don't allocate per-channel canvases for MCPPMT's 64+64
    // channels).
    //
    // Bound by min(Ceren, Scint, allocated per-tower canvases). The allocated
    // count is fCanvasFull.size() - 1 (slot 0 is the main heatmap canvas) and
    // init_Generic() already sizes it to min(C, S), but we keep this defensive
    // since a future caller may push extra canvases.
    if (fCaseName == "full") {
      const size_t nTowerCanvas = fCanvasFull.size() > 1 ? fCanvasFull.size() - 1 : 0;
      const size_t nTowers = std::min({fPlotter_Ceren.size(), fPlotter_Scint.size(), nTowerCanvas});
      for (size_t idx = 0; idx < nTowers; ++idx) {
        const size_t iTower = idx + 1;

        fCanvasFull.at(iTower)->cd(1);
        fPlotter_Ceren.at(idx).hist1D->Draw("Hist");

        fCanvasFull.at(iTower)->cd(2);
        fPlotter_Scint.at(idx).hist1D->Draw("Hist");
      }
    }
  } else if (fCaseName == "module") {
    // Per-tower distribution grid. Pad index for (info.row, info.col) in a
    // Divide(gridX, gridY) canvas (left-to-right, top-to-bottom, 1-based):
    //   pad = (gridY - info.col) * gridX + info.row
    // info.row/info.col were already remapped to local (1..gridX, 1..gridY)
    // coordinates in init_module(), so this works for both the KEK "all
    // towers" case and the CERN "one module's towers" case.
    for (size_t i = 0; i < fPlotter_Ceren.size(); ++i) {
      const auto& info = fPlotter_Ceren.at(i).info;
      if (info.row < 1 || info.row > fGridX_module ||
          info.col < 1 || info.col > fGridY_module) continue;
      const int pad = (fGridY_module - info.col) * fGridX_module + info.row;
      fCanvas->cd(pad);
      fPlotter_Ceren.at(i).hist1D->Draw("Hist");
    }
    for (size_t i = 0; i < fPlotter_Scint.size(); ++i) {
      const auto& info = fPlotter_Scint.at(i).info;
      if (info.row < 1 || info.row > fGridX_module ||
          info.col < 1 || info.col > fGridY_module) continue;
      const int pad = (fGridY_module - info.col) * fGridX_module + info.row;
      fCanvas->cd(pad);
      // "sames" overlays the Scint distribution on top of the Ceren one drawn
      // just above, sharing the pad and stats box positions.
      fPlotter_Scint.at(i).hist1D->Draw("Hist & sames");
    }
  }

  // if (fUsingAUX) gSystem->ProcessEvents();
  gSystem->Sleep(1000);
}

void TBplotengine::Update() {

  if (fCalcInfo == TBplotengine::CalcInfo::kAvgTimeStruc)
    for (int i = 0; i < fPlotter_Ceren.size(); i++)
      fPlotter_Ceren.at(i).hist1D->Scale(1./(float)fPlotter_Ceren.at(i).xInit);

  if (fCaseName == "single") {
    if (fCalcInfo == TBplotengine::CalcInfo::kIntADC || fCalcInfo == TBplotengine::CalcInfo::kPeakADC || fCalcInfo == TBplotengine::CalcInfo::kAvgTimeStruc)
      SetMaximum();

    fCanvas->cd();

    if (fCalcInfo == TBplotengine::CalcInfo::kOverlay) {
      fCanvas->cd();
      fPlotter_Ceren.at(0).hist2D->Draw("colz");

    } else {
      fCanvas->cd();
      fMainFrame->Draw();

      double stat_height = (1. - 0.2) / (double)fPlotter_Ceren.size();
      for (int i = 0; i < fPlotter_Ceren.size(); i++) {
        fCanvas->cd();
        fPlotter_Ceren.at(i).hist1D->Draw("Hist & sames");

        if (fIsFirst) {

          if (fCalcInfo == TBplotengine::CalcInfo::kIntADC || fCalcInfo == TBplotengine::CalcInfo::kPeakADC) {
            fCanvas->Update();
            // TPaveStats* stat = (TPaveStats*)fCanvas->GetPrimitive("stats");
            TPaveStats* stat = (TPaveStats*)fPlotter_Ceren.at(i).hist1D->FindObject("stats");
            // stat->SetName(fPlotter_Ceren.at(i).hist1D->GetName() + (TString)"_stat");
            stat->SetTextColor(fPlotter_Ceren.at(i).hist1D->GetLineColor());
            stat->SetY2NDC(1. - stat_height * i);
            stat->SetY1NDC(1 - stat_height * (i + 1));
            stat->SaveStyle();
          }
        }
      }
      if (fIsFirst) fIsFirst = false;
      if (fCalcInfo == TBplotengine::CalcInfo::kAvgTimeStruc) {
        fCanvas->cd();
        fLeg->Draw("same");
      }
    }
  } else if (fCaseName == "full" || fCaseName == "heatmap") {

    const std::string cerenLabel = (fCaseName == "heatmap") ? " MCPPMT C - "
                                                            : " CERENKOV - ";
    const std::string scintLabel = (fCaseName == "heatmap") ? " MCPPMT S - "
                                                            : " SCINTILLATION - ";

    // Title entry-count is sampled from the first plotter, but only if the
    // list is non-empty (an empty list happens when init_Generic() found no
    // tower-like channels in the loaded mapping). Falling through with empty
    // lists used to throw out_of_range from .at(0).
    const int cerenEntries = fPlotter_Ceren.empty() ? 0 : (int)fPlotter_Ceren.front().hist1D->GetEntries();
    const int scintEntries = fPlotter_Scint.empty() ? 0 : (int)fPlotter_Scint.front().hist1D->GetEntries();

    f2DHistCeren->SetTitle((TString)"Run " + std::to_string(fRunNum) + cerenLabel + std::to_string(cerenEntries));
    for (int i = 0; i < fPlotter_Ceren.size(); i++)
      f2DHistCeren->SetBinContent(fPlotter_Ceren.at(i).info.row, fPlotter_Ceren.at(i).info.col, (int)fPlotter_Ceren.at(i).hist1D->GetMean());

    f2DHistScint->SetTitle((TString)"Run " + std::to_string(fRunNum) + scintLabel + std::to_string(scintEntries));
    for (int i = 0; i < fPlotter_Scint.size(); i++)
      f2DHistScint->SetBinContent(fPlotter_Scint.at(i).info.row, fPlotter_Scint.at(i).info.col, (int)fPlotter_Scint.at(i).hist1D->GetMean());

    fCanvasFull.at(0)->cd(1);
    f2DHistCeren->Draw("colz text");

    fCanvasFull.at(0)->cd(2);
    f2DHistScint->Draw("colz text");

    fCanvasFull.at(0)->Update();

    // Per-channel 1D canvases exist only for "full" (per-tower), not for
    // "heatmap" (MCPPMT has 64+64 channels which we deliberately do not split
    // into per-channel canvases). Bounds are the same as in Draw().
    if (fCaseName == "full") {
      const size_t nTowerCanvas = fCanvasFull.size() > 1 ? fCanvasFull.size() - 1 : 0;
      const size_t nTowers = std::min({fPlotter_Ceren.size(), fPlotter_Scint.size(), nTowerCanvas});
      for (size_t idx = 0; idx < nTowers; ++idx) {
        const size_t iTower = idx + 1;

        fCanvasFull.at(iTower)->cd(1);
        fPlotter_Ceren.at(idx).hist1D->Draw("Hist");

        fCanvasFull.at(iTower)->cd(2);
        fPlotter_Scint.at(idx).hist1D->Draw("Hist");

        fCanvasFull.at(iTower)->Update();
      }
    }

    if (fIsFirst)fIsFirst = false;
  } else if (fCaseName == "module") {
    // Refresh the grid: redraw each pad's distributions and update the canvas
    // title with the current entry count. Pad index logic matches Draw().
    const int cerenEntries = fPlotter_Ceren.empty() ? 0 : (int)fPlotter_Ceren.front().hist1D->GetEntries();
    const std::string moduleLabel = fModule.empty() ? std::string("All") : fModule;
    fCanvas->SetTitle((TString)("Run " + std::to_string(fRunNum) + " " + moduleLabel +
                                " - " + std::to_string(cerenEntries)));

    // Helper: pad index from a (row, col) plotter, returning -1 if the cell
    // sits outside the local grid (defensive — shouldn't happen because
    // init_module already filtered/offset coords).
    auto padFor = [this](const TButility::mod_info& info) -> int {
      if (info.row < 1 || info.row > fGridX_module ||
          info.col < 1 || info.col > fGridY_module) return -1;
      return (fGridY_module - info.col) * fGridX_module + info.row;
    };

    for (size_t i = 0; i < fPlotter_Ceren.size(); ++i) {
      const int pad = padFor(fPlotter_Ceren.at(i).info);
      if (pad < 0) continue;
      fCanvas->cd(pad);
      fPlotter_Ceren.at(i).hist1D->Draw("Hist");
    }
    for (size_t i = 0; i < fPlotter_Scint.size(); ++i) {
      const int pad = padFor(fPlotter_Scint.at(i).info);
      if (pad < 0) continue;
      fCanvas->cd(pad);
      // "sames" preserves the C stat box already drawn so we have two stacked
      // TPaveStats per pad (one per histogram); we reposition them just below.
      fPlotter_Scint.at(i).hist1D->Draw("Hist & sames");
    }

    // First call materializes the TPaveStats objects via canvas Update. On
    // the very first cycle the C and S stat boxes are both placed by ROOT at
    // the default (top-right) position and overlap. Reposition them once: C
    // sits on top in blue, S directly below in red, both narrower so the
    // bin-shape underneath is still visible. ROOT persists the NDC coords
    // with the histogram, so subsequent updates keep the layout for free.
    fCanvas->Update();

    if (fIsFirst) {
      auto place = [](TH1D* h, double y2, double y1, Color_t col) {
        if (!h) return;
        auto* stat = (TPaveStats*)h->FindObject("stats");
        if (!stat) return;
        stat->SetX1NDC(0.62);
        stat->SetX2NDC(0.98);
        stat->SetY1NDC(y1);
        stat->SetY2NDC(y2);
        stat->SetTextColor(col);
        stat->SaveStyle();
      };
      for (size_t i = 0; i < fPlotter_Ceren.size(); ++i) {
        const int pad = padFor(fPlotter_Ceren.at(i).info);
        if (pad < 0) continue;
        fCanvas->cd(pad);
        place(fPlotter_Ceren.at(i).hist1D, 0.99, 0.79, kBlue);
      }
      for (size_t i = 0; i < fPlotter_Scint.size(); ++i) {
        const int pad = padFor(fPlotter_Scint.at(i).info);
        if (pad < 0) continue;
        fCanvas->cd(pad);
        place(fPlotter_Scint.at(i).hist1D, 0.79, 0.59, kRed);
      }
      fCanvas->Update();
      fIsFirst = false;
    }
  }

  // fCanvas exists for single + module modes (each uses a single, possibly
  // subdivided, canvas). full / heatmap render onto fCanvasFull (already
  // updated in the case-specific block above), so touching fCanvas there
  // would dereference null.
  if (fCaseName == "single" || fCaseName == "module") {
    fCanvas->cd();
    fCanvas->Update();
    if (fDraw) fCanvas->Pad()->Draw();
  }

  // ── Output: atomic ROOT write + per-canvas JSON dump ───────────────────────
  // ROOT is written to <name>.tmp.root then renamed; JSONs are emitted by
  // WriteCanvasesAsJSON() (also atomic). This lets the web UI poll mid-run
  // without ever reading a half-written file. full/heatmap both publish via
  // fCanvasFull; single uses fCanvas.
  std::string basePrefix;
  if (fCaseName == "full") {
    basePrefix = "Run" + std::to_string(fRunNum) + "_" + fCaseName + "_" + fMethod;
    if (fAuxCut) basePrefix += "_AuxCut";
    TString output    = (TString)("./output/" + basePrefix + ".root");
    TString tmpOutput = (TString)("./output/" + basePrefix + ".tmp.root");
    {
      TFile outoutFile(tmpOutput, "RECREATE");
      outoutFile.cd();
      for (int i = 0; i < fCanvasFull.size(); i++)
        fCanvasFull.at(i)->Write();
      outoutFile.Close();
    }
    std::rename(tmpOutput.Data(), output.Data());
  } else if (fCaseName == "heatmap") {
    basePrefix = "Run" + std::to_string(fRunNum) + "_" + fCaseName + "_" + fMethod + "_" + fModule;
    if (fAuxCut) basePrefix += "_AuxCut";
    TString output    = (TString)("./output/" + basePrefix + ".root");
    TString tmpOutput = (TString)("./output/" + basePrefix + ".tmp.root");
    {
      TFile outoutFile(tmpOutput, "RECREATE");
      outoutFile.cd();
      for (int i = 0; i < fCanvasFull.size(); i++)
        fCanvasFull.at(i)->Write();
      outoutFile.Close();
    }
    std::rename(tmpOutput.Data(), output.Data());
  } else if (fCaseName == "module") {
    // Module mode uses fCanvas (one subdivided canvas). When --module is not
    // given (KEK case) we tag the file with "All" so the filename doesn't end
    // in a stray underscore.
    const std::string moduleTag = fModule.empty() ? std::string("All") : fModule;
    basePrefix = "Run" + std::to_string(fRunNum) + "_" + fCaseName + "_" + fMethod + "_" + moduleTag;
    if (fAuxCut) basePrefix += "_AuxCut";
    TString output    = (TString)("./output/" + basePrefix + ".root");
    TString tmpOutput = (TString)("./output/" + basePrefix + ".tmp.root");
    {
      TFile outoutFile(tmpOutput, "RECREATE");
      outoutFile.cd();
      fCanvas->Write();
      outoutFile.Close();
    }
    std::rename(tmpOutput.Data(), output.Data());
  } else {
    basePrefix = "Run" + std::to_string(fRunNum) + "_" + fCaseName + "_" + fMethod + "_" + fModule;
    if (fAuxCut) basePrefix += "_AuxCut";
    TString output    = (TString)("./output/" + basePrefix + ".root");
    TString tmpOutput = (TString)("./output/" + basePrefix + ".tmp.root");
    {
      TFile outoutFile(tmpOutput, "RECREATE");
      outoutFile.cd();
      fCanvas->Write();
      outoutFile.Close();
    }
    std::rename(tmpOutput.Data(), output.Data());
  }

  WriteCanvasesAsJSON("./output", basePrefix);

  // if (fUsingAUX) gSystem->ProcessEvents();
  // else           fApp->Run(false);

  // std::cout << fLive << " " << fUsingAUX << std::endl;

  if (fDraw) {
    if (fLive && fUsingAUX) gSystem->ProcessEvents();
    if (fLive && !fUsingAUX) gSystem->ProcessEvents();
    if (!fLive && fUsingAUX) gSystem->ProcessEvents();
    if (!fLive && !fUsingAUX) fApp->Run(false);
  }



  gSystem->Sleep(1000);

  if (fLive)
    if (fCalcInfo == TBplotengine::CalcInfo::kAvgTimeStruc)
      for (int i = 0; i < fPlotter_Ceren.size(); i++)
        fPlotter_Ceren.at(i).hist1D->Scale((float)fPlotter_Ceren.at(i).xInit);


}

void TBplotengine::SetMaximum() {

  float max = -999;
  for (int i = 0; i < fPlotter_Ceren.size(); i++) {
    if (max < fPlotter_Ceren.at(i).hist1D->GetMaximum()) {
      max = fPlotter_Ceren.at(i).hist1D->GetMaximum();
    }
  }

  fMainFrame->GetYaxis()->SetRangeUser(0., max * 1.2);
}

void TBplotengine::SaveAs(TString output = "")
{
  if (output == "")
    output = "./output/Run" + std::to_string(fRunNum) + "_" + fCaseName + "_" + fMethod + "_" + fModule + ".root";

  TFile* outoutFile = new TFile(output, "RECREATE");

  outoutFile->cd();
  if (fCaseName == "single") {
    if (fMethod == "Overlay") {
      for (int i = 0; i < fPlotter_Ceren.size(); i++)
        fPlotter_Ceren.at(i).hist2D->Write();
    } else {
      for (int i = 0; i < fPlotter_Ceren.size(); i++)
        fPlotter_Ceren.at(i).hist1D->Write();
    }
  }

  outoutFile->Close();
}

void TBplotengine::WriteCanvasesAsJSON(const std::string& outDir, const std::string& basePrefix)
{
  auto dump = [&](TCanvas* c) {
    if (!c) return;
    TString json = TBufferJSON::ToJSON(c);
    std::string canvasName = c->GetName();
    std::string finalPath = outDir + "/" + basePrefix + "_" + canvasName + ".json";
    std::string tmpPath   = finalPath + ".tmp";
    {
      std::ofstream ofs(tmpPath);
      if (!ofs) return;
      ofs << json.Data();
    }
    std::rename(tmpPath.c_str(), finalPath.c_str());
  };

  if (fCaseName == "full" || fCaseName == "heatmap") {
    for (auto* c : fCanvasFull) dump(c);
  } else {
    dump(fCanvas);
  }
}

std::vector<int> TBplotengine::GetUniqueMID() {
  if (fCaseName == "single") {
    return fUtility.GetUniqueMID(fCIDtoPlot_Ceren);
  } else if (fCaseName == "full" || fCaseName == "heatmap" || fCaseName == "module") {
    return fUtility.GetUniqueMID(fCIDtoPlot_Ceren, fCIDtoPlot_Scint);
  }

  return std::vector<int>{};
}
