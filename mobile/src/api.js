import * as ImagePicker from 'expo-image-picker';

// CRITICAL: Replace with your laptop's actual Wi-Fi IP address!
const API_BASE_URL = "http://192.168.1.14:8000"; 

export const uploadClothingItem = async () => {
  let result = await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ImagePicker.MediaTypeOptions.Images,
    allowsEditing: true,
    quality: 0.8,
  });

  if (!result.canceled) {
    const localUri = result.assets[0].uri;
    const filename = localUri.split('/').pop();
    const match = /\.(\w+)$/.exec(filename);
    const type = match ? `image/${match[1]}` : `image`;

    const formData = new FormData();
    formData.append('file', { uri: localUri, name: filename, type });

    console.log("Sending to AI Backend:", API_BASE_URL);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = await response.json();
      console.log("AI Response:", data);
      return data;
    } catch (error) {
      console.error("Network Error:", error);
    }
  }
};