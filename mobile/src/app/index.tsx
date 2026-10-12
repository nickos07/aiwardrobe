import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Button, Alert } from 'react-native';
// Notice the '../api' because api.js is one folder up in src/
import { uploadClothingItem } from '../api'; 

export default function HomeScreen() {
  const handleUpload = async () => {
    console.log("Opening camera roll...");
    const response = await uploadClothingItem();
    
    if (response && response.status === "success") {
      Alert.alert(
        "Added to Wardrobe!", 
        `AI detected a ${response.metadata.primary_color} ${response.metadata.item_type} (${response.metadata.category})`
      );
    } else {
      Alert.alert("Upload Failed", "Check your terminal logs and Wi-Fi IP address.");
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>AI Wardrobe Stylist</Text>
      <Text style={styles.subtitle}>Mobile Client v1.0</Text>
      <Button title="Upload Clothing Item" onPress={handleUpload} />
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 30,
  }
});