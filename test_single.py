import torch 
import torch.nn as nn
import numpy as np 

class RDClassification(nn.Module):
    def __init__(self, in_channels=1, num_class=5, numFrames=8):
        super(RDClassification, self).__init__()
        self.in_channels = in_channels
        self.num_class = num_class
        self.numFrames = numFrames

        self.norm = nn.LayerNorm(normalized_shape=[self.numFrames, self.in_channels, 70, 64])
        self.norm_ra = nn.LayerNorm(normalized_shape=[self.numFrames, self.in_channels, 70, 64])

        self.first_cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=(5,5), padding=(2,2)),
            #nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5, 5), dilation=(4, 1), padding=(8,2)),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU()
        )

        self.first_cnn_ra = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=(5,5), padding=(2,2)),
            #nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5, 5), dilation=(4, 1), padding=(8,2)),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU()
        )

        self.second_cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(5, 5), padding=(2,2)),
            #nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), dilation=(2, 1), padding=(4,2)),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU()
        )

        self.second_cnn_ra = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(5, 5), padding=(2,2)),
            #nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), dilation=(2, 1), padding=(4,2)),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU()
        )

        self.third_cnn = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), padding=(2,2)),
            #nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), dilation=(2, 1), padding=(4,2)),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU()
        )

        self.third_cnn_ra = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), padding=(2,2)),
            #nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), dilation=(2, 1), padding=(4,2)),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU()
        )

        self.fourth_cnn = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), padding=(2,2)),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU()
        )

        self.fourth_cnn_ra = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), padding=(2,2)),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(input_size=256, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False, dropout=0.15)
        self.lstm_ra = nn.LSTM(input_size=256, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False, dropout=0.15)
        
        self.fc_hidden = nn.Linear(in_features=128, out_features=32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(in_features=32, out_features=self.num_class)
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(in_features=32, out_features=2)
        self.fc2 = nn.Linear(in_features=32, out_features=2)
        self.fc3 = nn.Linear(in_features=32, out_features=2)
        self.fc4 = nn.Linear(in_features=32, out_features=2)
        self.fc5 = nn.Linear(in_features=32, out_features=2)

    def forward(self, rd, ra):
        out = rd.unsqueeze(2)
        out = self.norm(out)
        out = torch.reshape(out, [out.shape[0]*out.shape[1], out.shape[2], out.shape[3], out.shape[4]])
        
        out = self.first_cnn(out)
        out = self.second_cnn(out)
        out = self.third_cnn(out)
        out = self.fourth_cnn(out)

        out_ra = ra.unsqueeze(2)
        out_ra = self.norm_ra(out_ra)
        out_ra = torch.reshape(out_ra, [out_ra.shape[0]*out_ra.shape[1], out_ra.shape[2], out_ra.shape[3], out_ra.shape[4]])
        
        out_ra = self.first_cnn_ra(out_ra)
        out_ra = self.second_cnn_ra(out_ra)
        out_ra = self.third_cnn_ra(out_ra)
        out_ra = self.fourth_cnn_ra(out_ra)

        out_ra = torch.reshape(out_ra, [-1, self.numFrames, out_ra.shape[1], out_ra.shape[2], out_ra.shape[3]])
        out_ra = torch.reshape(out_ra, [out_ra.shape[0], out_ra.shape[1], out_ra.shape[2]*out_ra.shape[3]*out_ra.shape[4]])
        
        out = torch.reshape(out, [-1, self.numFrames, out.shape[1], out.shape[2], out.shape[3]])
        out = torch.reshape(out, [out.shape[0], out.shape[1], out.shape[2]*out.shape[3]*out.shape[4]])
        
        out, (h_n,c_n) = self.lstm(out)
        out = out[:, -1, :]

        out_ra, (h_n,c_n) = self.lstm_ra(out_ra)
        out_ra = out_ra[:, -1, :]
        
        out = torch.concat([out, out_ra], dim=1)
        
        out = self.relu(self.dropout(self.fc_hidden(out)))

        o = self.softmax(self.fc(out))
        out1 = self.softmax(self.fc1(out))
        out2 = self.softmax(self.fc2(out))
        out3 = self.softmax(self.fc3(out))
        out4 = self.softmax(self.fc4(out))
        out5 = self.softmax(self.fc5(out))
        
        return (o, out1, out2, out3, out4, out5)


def read_doppler_file(filename):
    data = np.load(filename, allow_pickle=True)
    
    doppler_fft = data['doppler_fft']
    azimuth_fft = data['azimuth_fft']

    range_doppler = np.swapaxes(10*np.log10((np.abs(doppler_fft)**2).sum(axis=1)), 1, 2)
    range_azimuth = np.swapaxes(10*np.log10((np.abs(azimuth_fft) ** 2).sum(axis=2)), 1, 2)

    return range_doppler, range_azimuth

def read_processed_file(filename):
    data = np.load(filename, allow_pickle=True).item()

    range_doppler = data['rd']
    range_azimuth = data['ra']
    return range_doppler, range_azimuth

def load_model(model_file, in_channels=1, num_class=5, numFrames=8):
    model = RDClassification(in_channels=in_channels, num_class=num_class, numFrames=numFrames)
    model.load_state_dict(torch.load(model_file))

    return model 

def test(filename, model_file):
    #range_doppler, range_azimuth = read_processed_file(filename)
    range_doppler, range_azimuth = read_doppler_file(filename)
    if range_doppler.shape[0] > 8:
        indices = np.arange(0, range_doppler.shape[0]-1, 1)
        selected_indices = np.sort(np.random.choice(indices, size=8-1, replace=False))
        selected_indices = np.append(selected_indices, range_doppler.shape[0]-1)
        range_doppler = range_doppler[selected_indices]
        range_azimuth = range_azimuth[selected_indices]
    
    model = load_model(model_file, in_channels=1, num_class=5, numFrames=8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    range_doppler = torch.from_numpy(range_doppler)
    range_azimuth = torch.from_numpy(range_azimuth)
    range_doppler = range_doppler.unsqueeze(0)
    range_azimuth = range_azimuth.unsqueeze(0)

    range_doppler = range_doppler.to(device, dtype=torch.float)
    range_azimuth = range_azimuth.to(device, dtype=torch.float)

    multi_class, presence, left_chest, right_chest, left_pocket, right_pocket = model(range_doppler, range_azimuth)
    
    full_dist = multi_class.detach().cpu().numpy()[0]
    left_chest_dist = left_chest.detach().cpu().numpy()[0]
    right_chest_dist = right_chest.detach().cpu().numpy()[0]
    left_pocket_dist = left_pocket.detach().cpu().numpy()[0]
    right_pocket_dist = right_pocket.detach().cpu().numpy()[0]
    
    return full_dist, left_chest_dist, right_chest_dist, left_pocket_dist, right_pocket_dist

if __name__ == "__main__":
    full_dist, left_chest_dist, right_chest_dist, left_pocket_dist, right_pocket_dist = \
        test(
            #filename="F:/ConcealedWeapon/data/Monday_processed/0.npy", 
            filename="F:/ConcealedWeapon/data/forward_triggered_frames-00000.pickle",
            model_file="F:/ConcealedWeapon/weights/rd1/test_rd104"
        )
    print(full_dist, left_chest_dist, right_chest_dist, left_pocket_dist, right_pocket_dist)
