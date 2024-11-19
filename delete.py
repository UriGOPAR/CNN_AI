import os
import cv2

def delete_images_in_folder(folder_path):
    # Verificar si la ruta existe
    if not os.path.isdir(folder_path):
        print("La ruta especificada no existe.")
        return
    
    # Obtener la lista de archivos en la carpeta
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No hay imágenes en la carpeta especificada.")
        return
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        # Leer y mostrar la imagen
        image = cv2.imread(image_path)
        if image is not None:
            cv2.imshow('Imagen', image)
            print(f"Presiona 's' para borrar {image_file}, 'n' para no borrar.")
            
            # Esperar la entrada del usuario
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'):
                os.remove(image_path)
                print(f"{image_file} ha sido borrado.")
            elif key == ord('n'):
                print(f"{image_file} no fue borrado.")
            else:
                print(f"Tecla no reconocida. {image_file} no fue borrado.")
            
            # Cerrar la ventana después de la decisión
            cv2.destroyAllWindows()
        else:
            print(f"Error al leer {image_file}")

if __name__ == "__main__":
    # Pedir al usuario la ruta de la carpeta
    folder_name = input("Introduce el nombre de la carpeta dentro de 'Mushrooms': ")
    folder_path = os.path.join("Mushrooms", folder_name)
    
    delete_images_in_folder(folder_path)
