import numpy as np
import torch
from homework4 import CNP
import matplotlib.pyplot as plt

def prepare_dataset(train_ratio=0.6, val_ratio=0.2):
    data = np.load("state_trajectories.npy")

    # Calculate the number of samples for each split
    num_samples = data.shape[0]

    T = np.linspace(0, 1, data.shape[1]).reshape(-1, 1)
    T = np.tile(T, (data.shape[0], 1, 1))
    data = np.concatenate((data, T), axis=2)

    train_size = int(num_samples * train_ratio)

    val_size = int(num_samples * val_ratio)
    val_start = train_size
    val_end = val_start + val_size
    # Split the dataset

    train_data = data[:train_size]
    val_data = data[val_start:val_end]
    test_data = data[val_end:]

    # save the split datasets
    np.save("train_data.npy", train_data)
    np.save("val_data.npy", val_data)
    np.save("test_data.npy", test_data)


def train():

    num_epochs = 100000
    learning_rate = 0.0003
    batch_size = 32
    max_timestep = 100

    d_x = 2
    d_y = 4

    # Create the model.
    model = CNP(in_shape=(d_x, d_y), hidden_size=128, num_hidden_layers=2)
    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_data = np.load("train_data.npy")
    val_data = np.load("val_data.npy")

    max_obs = 10
    val_losses = []
    train_losses = []

    for epoch in range(num_epochs):

        perm = np.random.permutation(len(train_data))
        batch = train_data[perm[:batch_size]]

        # Create the observation set
        obs = torch.zeros((batch_size, max_obs, d_x + d_y))
        obs_mask = torch.zeros((batch_size, max_obs))
        target = torch.zeros((batch_size, 1, d_x))
        target_truth = torch.zeros((batch_size, 1, d_y))
        
        for j in range(batch_size):
            num_obs = np.random.randint(1, max_obs + 1)
            obs[j, :num_obs, :] = torch.tensor(batch[j, :num_obs, :]).to(torch.float32)
            obs_mask[j, :num_obs] = 1
            tar_index = np.random.randint(max_timestep)
            target[j, 0, :] = torch.tensor(batch[j, tar_index, [0, -1]]).to(torch.float32)
            target_truth[j, 0, :] = torch.tensor(batch[j, tar_index, 1:-1]).to(torch.float32)

        loss = model.nll_loss(obs, target, target_truth, obs_mask, None)
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            #print(f"Epoch {epoch}, Loss: {loss.item()}")
            # validate the model
            val_loss = 0
            with torch.no_grad():   
                for j in range(len(val_data)):
                    num_obs = np.random.randint(1, max_obs + 1)
                    obs = torch.tensor(val_data[j, :num_obs, :]).to(torch.float32)
                    obs_mask = torch.ones((num_obs))

                    tar_idx = np.random.randint(max_timestep)
                    target = torch.tensor(val_data[j, [tar_idx], [0, -1]]).to(torch.float32)
                    target_truth = torch.tensor(val_data[j, [tar_idx], 1:-1]).to(torch.float32)

                    val_loss += model.nll_loss(obs.unsqueeze(0), target.unsqueeze(0).unsqueeze(0), target_truth.unsqueeze(0), obs_mask.unsqueeze(0), None).item()
            val_loss /= len(val_data)

            val_losses.append(val_loss)

            if val_loss == min(val_losses):
                torch.save(model.state_dict(), "cnmp_best_model.pth")
                print("Saving model at epoch", epoch)

    # save loss
    np.save("train_losses.npy", train_losses)

def plot_loss():
    loss = np.load("train_losses.npy")
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("train_loss.png")
    plt.close()

def test():

    test_data = np.load("test_data.npy")

    model = CNP(in_shape=(2, 4), hidden_size=128, num_hidden_layers=2)
    model.load_state_dict(torch.load("cnmp_best_model.pth"))
    model.eval()

    object_errors = np.zeros((len(test_data), 1))
    ee_errors = np.zeros((len(test_data), 1))

    for j in range(len(test_data)):

        num_obs = np.random.randint(1, 11)
        obs = torch.tensor(test_data[j, :num_obs, :]).to(torch.float32)
        obs_mask = torch.ones((num_obs))

        tar_idx = np.random.randint(100)
        target = torch.tensor(test_data[j, [tar_idx], [0, -1]]).to(torch.float32)
        target_truth = torch.tensor(test_data[j, [tar_idx], 1:-1]).to(torch.float32)

        pred_mean, _ = model(obs.unsqueeze(0), target.unsqueeze(0).unsqueeze(0), obs_mask.unsqueeze(0))
        
        ee_errors[j] = torch.mean((pred_mean[0][0][:2] - target_truth[0][:2])**2).item()
        object_errors[j] = torch.mean((pred_mean[0][0][2:] - target_truth[0][2:])**2).item()

    np.save("object_errors.npy", object_errors)
    np.save("ee_errors.npy", ee_errors)

def plot_test():
    
    object_errors = np.load("object_errors.npy")
    ee_errors = np.load("ee_errors.npy")

    # bar plot with two bars, 1 for object errors and 1 for ee errors, with standard deviation with caps and different colors
    plt.bar([0, 1], [np.mean(object_errors), np.mean(ee_errors)], yerr=[np.std(object_errors), np.std(ee_errors)], capsize=5, color=['blue', 'orange'])
    plt.xticks([0, 1], ['Object Error', 'End Effector Error'])
    plt.ylim(0, 0.0035)
    plt.ylabel("Error")
    plt.title("Test Errors")
    plt.savefig("test_errors.png")
    



if __name__ == "__main__":
    #prepare_dataset()
    #train()
    #plot_loss()
    test()
    plot_test()