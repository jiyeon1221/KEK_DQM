#ifndef TButility_h
#define TButility_h 1

#include <vector>
#include <algorithm>
#include <map>
#include <string>

#include "TBdetector.h"

#include <TChain.h>
#include <TString.h>
#include <TH2.h>
#include <TFile.h>

class TButility
{
public:
  TButility() {}
  TButility(std::string fMapping_) {
    LoadMapping(fMapping_);
  }
  ~TButility() {}

  struct mod_info {
    int isCeren;
    int row;
    int col;

    mod_info(int isCeren_, int row_, int col_)
    : isCeren(isCeren_), row(row_), col(col_)
    {}
  };

  void LoadMapping(const std::string &path);

  TBcid GetCID(TString name) const;
  std::string GetName(TBcid cid) const;
  mod_info GetInfo(TBcid cid) const;
  mod_info GetInfo(TString name) const;

  // Const view onto the full name -> DQM info map. Used by TBplotengine to
  // enumerate the loaded mapping for data-driven discovery (so --type full
  // doesn't have to hardcode a tower-name list and grid size).
  const std::map<TString, mod_info>& GetNameInfo() const { return mapping_NAME_INFO; }

  std::vector<int> GetUniqueMID(std::vector<TBcid> aCID);
  std::vector<int> GetUniqueMID(std::vector<TBcid> aCID_1, std::vector<TBcid> aCID_2);
  std::vector<int> GetUniqueMID(std::vector<int> vec_1, std::vector<int> vec_2);

private:
  std::map<TBcid, std::string> mapping_CID_NAME;
  std::map<TBcid, mod_info> mapping_CID_INFO;
  std::map<TString, TBcid> mapping_NAME_CID;
  std::map<TString, mod_info> mapping_NAME_INFO;

};

#endif
