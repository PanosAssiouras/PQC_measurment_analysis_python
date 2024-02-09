#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdio>




using namespace std;

TF1 *f1, *f2;
double finter(double *x, double*par) {
   return TMath::Abs(f1->EvalPar(x,par) - f2->EvalPar(x,par));
}


int sort_string_vector(vector<string> &stringVec)
{
 for (vector<string>::size_type i = 0; i != stringVec.size(); ++i)
    {
    // Sorting the string vector
    sort(stringVec.begin(), stringVec.end());
    // Ranged Based loops. This requires a C++11 Compiler also
    // If you don't have a C++11 Compiler you can use a standard
    // for loop to print your vector.

    cout << stringVec[i] << endl;

}
 return 0;
}
void Convert_TestDataFile_To_RootTree(TString TextDataName, TString RootDataName)
{
   TFile *f = new TFile(RootDataName,"RECREATE");
   TTree *T = new TTree("TRee","data from ascii file");
   TNtuple data("data","IV","Current:Voltage");
   cout<<TextDataName<<endl;
   cout<<"root"<<endl;
	 std:: ifstream inputFile(TextDataName);
	  std::string line="";
    getline(inputFile,line);
    getline(inputFile,line);
    getline(inputFile,line);
    getline(inputFile,line);
    getline(inputFile,line);
    double Voltage, Current ;
	  while(getline(inputFile,line)){
         if ( line[0]=='C' || line[0]=='D' || line[0]=='V'|| !line.length())
           {  continue; }
     else{

        sscanf(line.c_str(), "%lf %lf ", &Current, &Voltage);
    //    cout<<seperation<<","<<Cback<<endl;
        data.Fill(Current, Voltage);

  }

   }
   data.Write();
   T->Write();
   f->Write();
   f->Close();
}

int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }
    while ((dirp = readdir(dp)) != NULL) {
		if((string(dirp->d_name)=="Results.txt")||(string(dirp->d_name)=="TextToRoot.C"||(string(dirp->d_name)=="Cback@300V.txt"))
    ||(string(dirp->d_name)==".")||(string(dirp->d_name)==".."||string(dirp->d_name).find(".root")!=-1)){continue;}
        files.push_back(string(dirp->d_name));
      //  cout<<string(dirp->d_name)<<endl;
    }
    closedir(dp);
    return 0;
}


int CloverMetal()
{

  ofstream myfile;
  myfile.open ("Results.txt",ios::app);

  TCanvas *c1=new TCanvas("c1","IV",300,10,900,500);
  c1->SetGrid();

   TMultiGraph *mg1 = new TMultiGraph();
  //  mg1->SetMinimum(-10E-06);
  // mg1->SetMaximum();
   mg1->SetTitle("VDPStop : HPK_VPX33234-011_PSS_HM_EE : Left and Right");
//   gStyle->SetTitleFontSize(0.080);
   gStyle->SetTitleAlign(23);



   int markerstyle=20;
   int markercolor =2;


   TLegend *leg1 = new TLegend(0.75, 0.20, 0.95, 0.65);
  // leg-> SetNColumns(1);
   leg1->SetTextColor(kBlue);;
   leg1->SetTextAlign(12);
   leg1->SetTextAlign(12);
   leg1->SetNColumns(1);
   leg1->SetTextSize(0.030);

    string dir = string(".");  // Set the folder for search (".") for parent file
    vector<string> files = vector<string>(); // This vector will contain the names of each file in the folder after getdir function
    getdir(dir,files);

    sort_string_vector(files);

    cout<<files.size()<<endl;
        for (unsigned int k = 0; k<files.size();k++) {

            TGraph *gr1=new TGraph();

            size_t pos1= files[k].find(".root");
            if(pos1==-1){
                size_t pos= files[k].find(".txt");
                cout << files[k] << endl;
                if(pos!=-1){
                  string r=files[k];
                  TString converted_root;
                  r=r.replace(r.begin()+pos,r.end(),".root");
                  //char *root_name = new char[pos + 1];
                  TString root_name=r.c_str();
                  Convert_TestDataFile_To_RootTree(files[k],root_name);

          TFile* in_file=new TFile(root_name);
          double Voltage,Current;
          float* row_content;

           TNtuple* data=(TNtuple*) in_file->GetObjectChecked("data","TNtuple");
           for(int irow=0; irow<data->GetEntries();++irow)
           {   data->GetEntry(irow);
               row_content=data->GetArgs();
               Current=row_content[0];
               Voltage=row_content[1];
               cout<<"Voltage="<<Voltage<<"Curent="<<Current<<endl;
               gr1->SetPoint(irow,Current,Voltage);

         }

      gr1->SetMarkerStyle(markerstyle);
      gr1->SetMarkerColor(markercolor);
      gr1->SetLineColor(markercolor);
      gr1->SetMarkerSize(1.2);
      gr1->GetXaxis()->SetTitle("Voltage [V]");
      gr1->GetYaxis()->SetTitle("Current [A]");
      f1= new TF1("f1","pol1", -100E-03, 100E-03);
      gr1->Fit("f1","R");
      gr1->GetFunction("f1")->SetLineColor(markercolor);
      double Rsh;
      double slope=(gr1->GetFunction("f1")->GetParameter(1));
      Rsh=(TMath::Pi()/TMath::Log(2))*(slope);
      cout<<"RSh="<<Rsh<<endl;
      mg1->Add(gr1);
      leg1->AddEntry(gr1,files[k].substr(0,pos).c_str(),"lp");
      myfile<<files[k]<<"\t"<<Rsh<<endl;

      markerstyle++;
      markercolor++;
      if(markercolor==10){
      markercolor=30;
      }
      if(markercolor==0){
      markercolor=1;
      }
      if(markerstyle==49){
      markerstyle=20;
      }



      }
     }



     }

            myfile.close();


                c1->cd();
               // pad.SetTitle("Total Current Pixel sensors 50x50 with 150 thickness ");
                //pad.DrawFrame(0,0,55,1e-4);
              //    TGaxis::SetMaxDigits(3);
                  gPad->Modified();
                  gPad->Update();
                  mg1->Draw("apl");
                  mg1->GetXaxis()->SetTitle("Current [A]");
                  mg1->GetYaxis()->SetTitle("Voltage [V]");
                  //mg1->GetYaxis()->SetRange(0.0,60.0);
            //      mg1->GetXaxis()->SetTitleSize(0.050);
              //    mg1->GetYaxis()->SetTitleSize(0.050);
              //    mg1->GetXaxis()->SetLabelSize(0.045);
              //    mg1->GetYaxis()->SetLabelSize(0.045);
                //  mg1->GetYaxis()->SetNdivisions(508);
                //  mg1->GetXaxis()->SetNdivisions(512);
                //  mg1->GetXaxis()->SetLimits(0.0,55.0);
                  leg1->Draw("apl");
                  gPad->Modified();
                  gPad->Update();



    return 0;
}
