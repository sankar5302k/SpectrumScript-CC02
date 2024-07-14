import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb



class ImageColorAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.dominant_color = None
        self.second_dominant_color = None
        self.palette_colors = None
        self.matching_colors = None

    def count_no_of_colors(self, image_path):
        image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        pixels = image_rgb.reshape(-1, 3)
        colours, counts = np.unique(pixels, axis=0, return_counts=True)
        return len(colours)

    def preprocess_image(self, image, resize_width=100):
        height, width = image.shape[:2]
        new_height = int((resize_width / width) * height)
        resized_image = cv2.resize(image, (resize_width, new_height))
        return resized_image

    def create_mask(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        return mask

    def get_dominant_color(self, image, k=4):
        resized_image = self.preprocess_image(image)
        mask = self.create_mask(resized_image)
        masked_pixels = resized_image[mask == 255]
        pixels = masked_pixels.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        label_counts = np.bincount(labels)
        dominant_color = cluster_centers[label_counts.argmax()]
        return dominant_color

    def get_second_dominant_color(self, image, k=4):
        resized_image = self.preprocess_image(image)
        mask = self.create_mask(resized_image)
        masked_pixels = resized_image[mask == 255]
        pixels = masked_pixels.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)
        cluster_centers_sorted = sorted(kmeans.cluster_centers_, key=lambda x: np.count_nonzero(kmeans.labels_ == kmeans.predict([x])), reverse=True)
        second_dominant_color = cluster_centers_sorted[1] if len(cluster_centers_sorted) > 1 else cluster_centers_sorted[0]
        return second_dominant_color

    def increase_brightness(self, hsv_color, increment):
        new_value = min(1.0, hsv_color[2] + increment)
        return hsv_to_rgb([hsv_color[0], hsv_color[1], new_value]) * 255.0

    def suggest_matching_colors(self, dominant_color, brightness_increment=0.4):
        dominant_hsv = rgb_to_hsv(dominant_color.reshape(1, 1, 3) / 255.0)[0][0]
        complementary_hue = (dominant_hsv[0] + 0.5) % 1.0
        complementary_hsv = np.array([complementary_hue, dominant_hsv[1], dominant_hsv[2]])
        complementary_color = self.increase_brightness(complementary_hsv, brightness_increment)
        triadic_hues = [(dominant_hsv[0] + 1/3) % 1.0, (dominant_hsv[0] + 2/3) % 1.0]
        triadic_colors = [self.increase_brightness(np.array([hue, dominant_hsv[1], dominant_hsv[2]]), brightness_increment) for hue in triadic_hues]
        analogous_hues = [(dominant_hsv[0] + 1/12) % 1.0, (dominant_hsv[0] - 1/12) % 1.0]
        analogous_colors = [self.increase_brightness(np.array([hue, dominant_hsv[1], dominant_hsv[2]]), brightness_increment) for hue in analogous_hues]
        split_complementary_hues = [(dominant_hsv[0] + 5/12) % 1.0, (dominant_hsv[0] - 5/12) % 1.0]
        split_complementary_colors = [self.increase_brightness(np.array([hue, dominant_hsv[1], dominant_hsv[2]]), brightness_increment) for hue in split_complementary_hues]
        suggested_colors = [complementary_color] + triadic_colors + analogous_colors + split_complementary_colors
        suggested_colors = np.array(suggested_colors, dtype=int)
        return suggested_colors

    def get_palette_colors(self, image, k=9):
        resized_image = self.preprocess_image(image)
        mask = self.create_mask(resized_image)
        masked_pixels = resized_image[mask == 255]
        if np.all(masked_pixels == 0):
            return np.array([[0, 0, 0]])
        pixels = masked_pixels.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k,n_init=10, random_state=0)
        kmeans.fit(pixels)
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        label_counts = np.bincount(labels)
        sorted_indices = np.argsort(label_counts)[::-1]
        sorted_colors = cluster_centers[sorted_indices]
        return sorted_colors

    def rgb_to_hex(self, r, g, b):
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    def adjust_brightness(self, rgb_color, factor, target='none'):
        rgb_color = np.array(rgb_color, dtype=np.float32)
        rgb_color = rgb_color.reshape(1, 1, 3) / 255.0
        hsv_color = rgb_to_hsv(rgb_color)[0][0]
        if target == 'fade_to_white':
            new_saturation = np.clip(hsv_color[1] * (1 - factor), 0, 1)
            new_value = np.clip(hsv_color[2] + (1 - hsv_color[2]) * factor, 0, 1)
        elif target == 'darken_to_black':
            new_saturation = hsv_color[1]
            new_value = np.clip(hsv_color[2] * (1 - factor), 0, 1)
        else:
            new_saturation = hsv_color[1]
            new_value = np.clip(hsv_color[2] * factor, 0, 1)
        new_color = hsv_to_rgb([hsv_color[0], new_saturation, new_value]) * 255.0
        return new_color.astype(int)

    def process_color(self, rgb_colors):
        light_dark=[]
        for rgb_color in rgb_colors:
            some=[]
            faded_color = self.adjust_brightness(rgb_color, 0.37, 'fade_to_white')
            faded_faded_color = self.adjust_brightness(faded_color, 0.35, 'fade_to_white')
            darkened_color = self.adjust_brightness(rgb_color, 0.1, 'darken_to_black')
            darkened_darkened_color = self.adjust_brightness(darkened_color, 0.1, 'darken_to_black')
            some=[faded_faded_color, faded_color, rgb_color, darkened_color, darkened_darkened_color]
            light_dark.append(list(some))
        return light_dark
    
    def calculate_color_percentage(self, color_RGB_list, image):
        def is_grayscale(image):
            if len(image.shape) == 2:
                return True
            elif image.shape[2] == 1:
                return True
            elif np.all(image[:, :, 0] == image[:, :, 1]) and np.all(image[:, :, 1] == image[:, :, 2]):
                return True
            return False
        #for grayscale image
        if is_grayscale(image):
            total_pixels = image.shape[0] * image.shape[1]
            if np.array_equal(image, image[0, 0]):
                return [100] + [0] * (len(color_RGB_list) - 1)
            else:
               hist = cv2.calcHist([image], [0], None, [256], [0, 256])
               num_palette_colors = min(10, len(hist))
               total_pixels = image.shape[0] * image.shape[1]
               palette_colors = []
               percentages = []
               for i in range(num_palette_colors):
                   intensity = np.argmax(hist)
                   if hist[intensity] > 0:
                       palette_colors.append(intensity)
                       percentage = (hist[intensity] / total_pixels) * 100
                       percentages.append(percentage)
                   hist[intensity] = 0
               percentages = np.array(percentages)
               percentages = (percentages / np.sum(percentages)) * 100
               percentages=percentages.flatten().tolist()
               for i in range(len(percentages)):
                   percentages[i]=round(percentages[i])
               return percentages
        #for non-grayscale image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        total_pixels = image.shape[0] * image.shape[1]
        color_percentages = []
        for color_RGB in color_RGB_list:
            color_HSV = cv2.cvtColor(np.uint8([[color_RGB]]), cv2.COLOR_RGB2HSV)[0][0]
            hue = color_HSV[0].astype(np.int32)
            lower_hue = max(0, hue - 10)
            upper_hue = min(179, hue + 10)
            if upper_hue < lower_hue:
                lower_range1 = np.array([lower_hue, 50, 50])
                upper_range1 = np.array([179, 255, 255])
                lower_range2 = np.array([0, 50, 50])
                upper_range2 = np.array([upper_hue, 255, 255])
                mask = cv2.inRange(hsv_image, lower_range1, upper_range1) | cv2.inRange(hsv_image, lower_range2, upper_range2)
            else:
                lower_range = np.array([lower_hue, 50, 50])
                upper_range = np.array([upper_hue, 255, 255])
                mask = cv2.inRange(hsv_image, lower_range, upper_range)
            color_pixels = np.sum(mask > 0)
            color_percentages.append(color_pixels / total_pixels * 100)
        total_percentage = sum(color_percentages)
        if total_percentage == 0:
            return [0] * len(color_RGB_list)
        normalized_percentages = [(p / total_percentage) * 100 for p in color_percentages]
        rounded_percentages = [round(p) for p in normalized_percentages]
        difference = 100.0 - sum(rounded_percentages)
        for i in range(len(rounded_percentages)):
            if difference == 0:
                break
            rounded_percentages[i] += round(difference)
            difference = 100.0 - sum(rounded_percentages)
        sorted_percentages = sorted(rounded_percentages, reverse=True)
        return sorted_percentages
    
    
    def analyze_image(self):
        try:
            self.images = cv2.imread(self.image_path)
            
            if self.images is None:
                raise ValueError(f"Image not found: {self.image_path}")
            
            self.count=self.count_no_of_colors(self.images)
            self.image = cv2.cvtColor(self.images, cv2.COLOR_BGR2RGB)
            
            if np.all(self.image == 0):
                self.palette_colors = np.array([[0, 0, 0]])
                self.dominant_color=self.palette_colors[0]
                self.second_dominant_color=self.palette_colors[0]
            elif np.all(self.image == 255):
                self.palette_colors = np.array([[255, 255, 255]])
                self.dominant_color=self.palette_colors[0]
                self.second_dominant_color=self.palette_colors[0]
            else:
                if self.count>10:
                    self.palette_colors = self.get_palette_colors(self.image)
                    if self.count==1:
                        self.dominant_color = self.palette_colors[0]
                        self.second_dominant_color = self.palette_colors[0]
                    else:
                        self.dominant_color = self.palette_colors[0]
                        self.second_dominant_color = self.palette_colors[1]
                else:
                    self.palette_colors = self.get_palette_colors(self.image,self.count)
                    if self.count==1:
                        self.dominant_color = self.palette_colors[0]
                        self.second_dominant_color = self.palette_colors[0]
                    else:
                        self.dominant_color = self.palette_colors[0]
                        self.second_dominant_color = self.palette_colors[1]
                
            self.matching_colors = self.suggest_matching_colors(self.dominant_color)

            all_colors = list(self.palette_colors)
            all_palette_colors=all_colors
            for some in range(len(all_palette_colors)):
                all_palette_colors[some]=all_palette_colors[some].astype(int)
           
            lighter_darker_versions = self.process_color(all_palette_colors)
            
            percentages = []
            percentages = self.calculate_color_percentage(all_palette_colors, self.images)
            pr=percentages
            dom_color=self.rgb_to_hex(*self.dominant_color.astype(int))
            dominant_versions = lighter_darker_versions[0]
            hex_dominant_versions = [self.rgb_to_hex(*color.astype(int)) for color in dominant_versions]
            
            dom2_color=self.rgb_to_hex(*self.second_dominant_color.astype(int))
            if len(lighter_darker_versions)==1:
                second_dominant_versions = lighter_darker_versions[0]
            else:
                second_dominant_versions = lighter_darker_versions[1]
            hex_second_dominant_versions = [self.rgb_to_hex(*color.astype(int)) for color in second_dominant_versions]
            all_colors_hex=[]
            for color in all_colors:
                abc=self.rgb_to_hex(*color.astype(int))
                all_colors_hex.append(abc)
            hex_versions=[]
            for versions in lighter_darker_versions:
                xyz = [self.rgb_to_hex(*color.astype(int)) for color in versions]
                hex_versions.append(xyz)
            matching_colors_hex=[]
            for color in self.matching_colors:
                cba=self.rgb_to_hex(*color.astype(int))
                matching_colors_hex.append(cba)
            if self.count<=10:
                k_value=self.count
            else:
                k_value=10
            
            everything=[dom_color,dom2_color] 
            everything.insert(1,list(hex_dominant_versions))
            everything.insert(3,list(hex_second_dominant_versions))
            everything.append(list(all_colors_hex))
            everything.append(list(hex_versions))
            everything.append(list(matching_colors_hex))
            everything.append(list(pr))
            everything.append(k_value)
            return everything

        except Exception as e:
            print("Error:", e)

