using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Threading;
using Newtonsoft.Json.Serialization;
using Newtonsoft.Json;

namespace NeuroPlayClient
{
    public partial class mainForm : Form
    {
        int i = 0;
        public mainForm()
        {
            InitializeComponent();
        }

        private void search_Click(object sender, EventArgs e)
        {
            WebRequest request = WebRequest.Create("http://127.0.0.1:2336/startSearch");
            WebResponse response = request.GetResponse();
            statusLabel.Text = ((HttpWebResponse)response).StatusDescription;
        }

        private void grabData_Click(object sender, EventArgs e)
        {
            timer.Interval = 4000;
            timer.Start();
        }

        private void disconnect_Click(object sender, EventArgs e)
        {
            timer.Stop();

            WebRequest request = WebRequest.Create("http://127.0.0.1:2336/disableDataGrabMode");
            WebResponse response = request.GetResponse();
            statusLabel.Text = ((HttpWebResponse)response).StatusDescription;

            request = WebRequest.Create("http://127.0.0.1:2336/startSearch");
            response = request.GetResponse();
            statusLabel.Text = ((HttpWebResponse)response).StatusDescription;

            Thread.Sleep(500);

            request = WebRequest.Create("http://127.0.0.1:2336/stopSearch");
            response = request.GetResponse();
            statusLabel.Text = ((HttpWebResponse)response).StatusDescription;

        }

        private void connectButton_Click(object sender, EventArgs e)
        {
            WebRequest request = WebRequest.Create("http://127.0.0.1:2336/startDevice?id=0");
            WebResponse response = request.GetResponse();
            statusLabel.Text = ((HttpWebResponse)response).StatusDescription;

            Thread.Sleep(500);

            request = WebRequest.Create("http://127.0.0.1:2336/enableDataGrabMode");
            response = request.GetResponse();
            statusLabel.Text = ((HttpWebResponse)response).StatusDescription;     

            request = WebRequest.Create("http://127.0.0.1:2336/setDataStorageTime?value=3");
            response = request.GetResponse();
            statusLabel.Text = ((HttpWebResponse)response).StatusDescription;
        }

        private void timer_Tick(object sender, EventArgs e)
        {
            WebRequest request = WebRequest.Create("http://127.0.0.1:2336/grabFilteredData");
            WebResponse response = request.GetResponse();
            statusLabel.Text = ((HttpWebResponse)response).StatusDescription;

            Stream stream;
            stream = response.GetResponseStream();
            StreamReader reader = new StreamReader(stream);

            string data = reader.ReadLine();
            dataBox.Text = data;

            string directiry = Directory.GetCurrentDirectory();
            string fileName = "line" + i + ".json";
            string fullFileName = Path.Combine(directiry, fileName);
            File.WriteAllText(fullFileName, data);

            i++;
        }
    }
}




//{ "command":"grabfiltereddata","data":[[182.04,44.58,-30.8,-21.05,-194.77,-91.99,146.15,173.54,80.93,-6.22,6.3,46.03,-85.56,-151.97,145.9,214.44],[-14.57,-31,128.64,199.72,112.11,-24.58,-116.25,15.18,138.91,-168.71,-276.22,107.99,59.89,-147.55,200.56,192.89],[-39.73,140.15,-145.74,-151.12,145.27,230.45,109.98,-70.87,-144.24,31.74,272.98,219.81,-77.61,-15.11,211.95,-88.53],[-123.04,-127.32,16.54,252.53,122.22,-232.84,-96.98,306.59,227.69,-143.1,-97.72,157.35,2.7,-201.36,19.62,68.67],[181.34,-163.24,-83.29,155.52,-40.84,-55.74,152.01,-18.5,-66.41,169.61,108.65,-4.39,94.01,241.12,263,63.37],[-24.6,55.65,262.73,83.53,-82.88,-146.47,-223.76,9.47,249.25,118.96,-0.36,24.66,19.95,-88.79,-202.48,-25.16],[-21.04,237.65,126.83,-133.47,-57.51,108.87,8.4,-181.6,-143.84,67.32,83.11,-96.92,-51.65,26.85,-41.68,4.31],[211.31,-131.63,-222.32,2.07,38.8,-69.43,-148.84,-146.86,135.08,275.09,42.71,22.04,99.31,-55.17,-118.79,-101.75]],"result":true,"time":"2021.11.20-04.25.16.275"}
