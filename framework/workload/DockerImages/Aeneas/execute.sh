#!/bin/bash
cd Aeneas/aeneas
for i in {1..500}
	do
	python -m aeneas.tools.execute_task ~/assets/audio/p001.mp3 ~/assets/text/p001.xhtml "task_language=eng|os_task_file_format=smil|os_task_file_smil_audio_ref=audio.mp3|os_task_file_smil_page_ref=page.xhtml|is_text_type=unparsed|is_text_unparsed_id_regex=f[0-9]+|is_text_unparsed_id_sort=numeric" map.smil
done
