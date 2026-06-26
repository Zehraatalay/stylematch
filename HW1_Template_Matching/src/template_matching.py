from pathlib import Path

import cv2

def load_templates(template_dir):
    template_paths = sorted(Path(template_dir).glob("*.png"))

    templates = []

    for path in template_paths:
        image = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)


        if image is None:
            raise ValueError(f"Template could not be read: {path}")
        
        templates.append(
            {
                "path" : str(path),
                "image" : image
            }
        )
    return templates

def resize_template(template,scale):
    height,width = template.shape[:2]


    new_width = int(width * scale)
    new_height = int(height * scale)



    if new_width  < 8 or new_height < 8:
        return None    


    return cv2.resize(template,(new_width,new_height), interpolation=cv2.INTER_AREA)


def match_single_template(image,template,scales):
    best_score = -1
    best_bbox = None


    image_height,image_width = image.shape[:2]



    for scale in scales:
        resized_template = resize_template(template,scale)

        if resized_template is None:
            continue

        template_height, template_width = resized_template.shape[:2]

        if template_height > image_height or template_width > image_width:
            continue


        result = cv2.matchTemplate(
            image,
            resized_template,
            cv2.TM_CCOEFF_NORMED
        )


        _,max_score, _, max_location = cv2.minMaxLoc(result)


        if max_score > best_score:
            x, y = max_location
            best_score = float(max_score)
            best_bbox = [ x,y,x + template_width,y + template_height]
    return best_score,best_bbox



def detect_with_templates(image,templates,threshold):
    scales = [1.0,0.75,0.5,0.375,0.25]


    best_score = -1
    best_bbox = None
    best_template_path = None


    for template in templates:
       score,bbox = match_single_template(
           image= image,
           template = template["image"],
           scales=scales
       )


       if score > best_score:
          best_score = score
          best_bbox = bbox
          best_template_path = template["path"]

    detected = best_score >= threshold
    
    if not detected:
       best_bbox = None
    return {
        "detected": detected,
        "score": best_score,
        "bbox": best_bbox,
        "template_path": best_template_path
    }