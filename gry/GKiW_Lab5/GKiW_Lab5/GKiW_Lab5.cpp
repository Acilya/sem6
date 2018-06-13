#include "stdafx.h"

#pragma region Zmienne globalne

SCameraState player;

double T = 0.0;

int mouseX = 0;
int mouseY = 0;

bool captureMouse = true;
bool free3DMovement = false;

float mouseSensitivity = .15f;

// Zmienne przechowuj1ce identyfikatory tekstur
GLuint chest_texture;

#pragma endregion

// Funkcja odczytuj1ca bitmape i tworz1ca na jej podstawie teksture z zadanym rodzajem filtracji
GLuint LoadTexture(char * file, int magFilter, int minFilter) {
	
	// Odczytanie bitmapy
	Bitmap *tex = new Bitmap();
	if (!tex->loadBMP(file)) {
		printf("ERROR: Cannot read texture file \"%s\".\n", file);
		return -1;
	}

	// Utworzenie nowego id wolnej tekstury
	GLuint texId;
	glGenTextures(1, &texId);

	// "Bindowanie" tekstury o nowoutworzonym id
	glBindTexture(GL_TEXTURE_2D, texId);

	// Okreolenie parametrów filtracji dla tekstury
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter); // Filtracja, gdy tekstura jest powiekszana
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter); // Filtracja, gdy tekstura jest pomniejszana

	// Wys3anie tekstury do pamieci karty graficznej zale?nie od tego, czy chcemy korzystaa z mipmap czy nie
	if (minFilter == GL_LINEAR_MIPMAP_LINEAR || minFilter == GL_LINEAR_MIPMAP_NEAREST) {
		// Automatyczne zbudowanie mipmap i wys3anie tekstury do pamieci karty graficznej
		gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, tex->width, tex->height, GL_RGB, GL_UNSIGNED_BYTE, tex->data);
	}
	else {
		// Wys3anie tekstury do pamieci karty graficznej 
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex->width, tex->height, 0, GL_RGB, GL_UNSIGNED_BYTE, tex->data);
	}
	
	// Zwolnienie pamieci, usuniecie bitmapy z pamieci - bitmapa jest ju? w pamieci karty graficznej
	delete tex;
	
	// Zwrócenie id tekstury
	return texId;
}

// Funkcja, która odpowiada za za3adowanie tekstur
void LoadTextures() {
	chest_texture = LoadTexture("Resources\\Chest.bmp", GL_NEAREST, GL_NEAREST);
}

int main(int argc, char* argv[])
{
	glutInit(&argc, argv);

	glutInitWindowPosition(100, 100);
	glutInitWindowSize(640, 360);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);

	glutCreateWindow("GKiW: Lab 5");

	glutDisplayFunc(OnRender);
	glutReshapeFunc(OnReshape);
	glutKeyboardFunc(OnKeyPress);
	glutKeyboardUpFunc(OnKeyUp);
	glutPassiveMotionFunc(OnMouseMove);
	glutMotionFunc(OnMouseMove);
	glutTimerFunc(17, OnTimer, 0);

	glEnable(GL_DEPTH_TEST);

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);
	
	float gl_amb[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, gl_amb);

	player.pos.x = 0.0f;
	player.pos.y = 1.0f;
	player.pos.z = 4.0f;

	player.dir.x = 0.0f;
	player.dir.y = 0.0f;
	player.dir.z = -1.0f;

	player.speed = .4f;

	glutWarpPointer(glutGet(GLUT_WINDOW_WIDTH) / 2, glutGet(GLUT_WINDOW_HEIGHT) / 2);
	mouseX = glutGet(GLUT_WINDOW_WIDTH) / 2;
	mouseY = glutGet(GLUT_WINDOW_HEIGHT) / 2;
	glutSetCursor(GLUT_CURSOR_NONE);

	LoadTextures(); // Ka?d1 teksture 3adujemy *raz* (nie w ka?dej klatce!), np. przed wejociem do petli g3ównej

	glutMainLoop();

	return 0;
}

#pragma region Obsluga wejscia

bool keystate[255];

void OnKeyPress(unsigned char key, int x, int y) {
	if (!keystate[key]) {
		keystate[key] = true;
		OnKeyDown(key, x, y);
	}
}

void OnKeyDown(unsigned char key, int x, int y) {
	if (key == 27) {
		glutLeaveMainLoop();
	}
	if (key == 'm' || key == 'M') {
		if (captureMouse) {
			captureMouse = false;
			glutSetCursor(GLUT_CURSOR_LEFT_ARROW);
		}
		else {
			captureMouse = true;
			glutWarpPointer(glutGet(GLUT_WINDOW_WIDTH) / 2, glutGet(GLUT_WINDOW_HEIGHT) / 2);
			glutSetCursor(GLUT_CURSOR_NONE);
		}
	}
	if (key == 'l' || key == 'L') {
		free3DMovement = !free3DMovement;
	}
}

void OnKeyUp(unsigned char key, int x, int y) {
	keystate[key] = false;
}

void OnMouseMove(int x, int y) {
	mouseX = x;
	mouseY = y;
}

#pragma endregion

void OnTimer(int id) {

	glutTimerFunc(17, OnTimer, 0);

	#pragma region Ruch kamery

	if (captureMouse) {
		player.velRY = -mouseSensitivity * (glutGet(GLUT_WINDOW_WIDTH) / 2 - mouseX);
		player.velRX = mouseSensitivity * (glutGet(GLUT_WINDOW_HEIGHT) / 2 - mouseY);
		glutWarpPointer(glutGet(GLUT_WINDOW_WIDTH) / 2, glutGet(GLUT_WINDOW_HEIGHT) / 2);
	}

	if (keystate['w']) {
		player.velM = player.speed;
	}
	if (keystate['s']) {
		player.velM = -player.speed;
	}
	if (keystate['a']) {
		player.velS = -player.speed;
	}
	if (keystate['d']) {
		player.velS = player.speed;
	}
	if (keystate['q']) {
		player.velRY = -player.speed;
	}
	if (keystate['e']) {
		player.velRY = player.speed;
	}
	if (keystate['f']) {
		player.velRX = -player.speed;
	}
	if (keystate['c']) {
		player.velRX = player.speed;
	}

	float T = acos(player.dir.y);
	float G = atan2(player.dir.z, player.dir.x);
	T -= player.velRX * .03f;
	G += player.velRY * .03f;
	player.dir.x = sin(T) * cos(G);
	player.dir.y = cos(T);
	player.dir.z = sin(T) * sin(G);

	vec3 per;
	per.x = -player.dir.z;
	per.y = 0;
	per.z = player.dir.x;

	player.pos.x += player.dir.x * player.velM * .1f;
	if (free3DMovement) {
		player.pos.y += player.dir.y * player.velM * .1f;
	}
	else {
		player.pos.y = 1.0f;
	}
	player.pos.z += player.dir.z * player.velM * .1f;

	player.pos.x += per.x * player.velS * .1f;
	if (free3DMovement) {
		player.pos.y += player.dir.y * player.velM * .1f;
	}
	else {
		player.pos.y = 1.0f;
	}
	player.pos.z += per.z * player.velS * .1f;

	player.velRX /= 1.2;
	player.velRY /= 1.2;
	player.velM /= 1.2;
	player.velS /= 1.2;

	#pragma endregion
}

void OnRender() {

	T = glutGet(GLUT_ELAPSED_TIME);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	gluLookAt(
		player.pos.x, player.pos.y, player.pos.z,
		player.pos.x + player.dir.x, player.pos.y + player.dir.y, player.pos.z + player.dir.z,
		0.0f, 1.0f, 0.0f
	);

	#pragma region Swiatlo
	
		float l0_amb[] = { 0.2f, 0.2f, 0.2f, 1.0f };
		float l0_dif[] = { 1.0f, 1.0f, 1.0f, 1.0f };
		float l0_spe[] = { 1.0f, 1.0f, 1.0f, 1.0f };
		float l0_pos[] = { 1.0f, 5.0f, 4.0f, 1.0f };
		glLightfv(GL_LIGHT0, GL_AMBIENT, l0_amb);
		glLightfv(GL_LIGHT0, GL_DIFFUSE, l0_dif);
		glLightfv(GL_LIGHT0, GL_SPECULAR, l0_spe);
		glLightfv(GL_LIGHT0, GL_POSITION, l0_pos);

	#pragma endregion

	#pragma region Rysowanie scian

		glBegin(GL_QUADS);

			#pragma region Przednia sciana
			{
				float m_amb[] = { 1.0f, 1.0f, 1.0f, 1.0f };
				float m_dif[] = { 1.0f, 1.0f, 1.0f, 1.0f };
				float m_spe[] = { 0.0f, 0.0f, 0.0f, 1.0f };
				glMaterialfv(GL_FRONT, GL_AMBIENT, m_amb);
				glMaterialfv(GL_FRONT, GL_DIFFUSE, m_dif);
				glMaterialfv(GL_FRONT, GL_SPECULAR, m_spe);
			
				glNormal3f( 0.0f,  0.0f,  1.0f);
				glVertex3f(-5.0f,  5.0f, -5.0f);

				glNormal3f( 0.0f,  0.0f,  1.0f);
				glVertex3f(-5.0f,  0.0f, -5.0f);

				glNormal3f( 0.0f,  0.0f,  1.0f);
				glVertex3f( 5.0f,  0.0f, -5.0f);

				glNormal3f( 0.0f,  0.0f,  1.0f);
				glVertex3f( 5.0f,  5.0f, -5.0f);
			}
			#pragma endregion

			#pragma region Lewa sciana
			{
				float m_amb[] = { 1.0f, 0.0f, 0.0f, 1.0f };
				float m_dif[] = { 1.0f, 0.0f, 0.0f, 1.0f };
				float m_spe[] = { 0.0f, 0.0f, 0.0f, 1.0f };
				glMaterialfv(GL_FRONT, GL_AMBIENT, m_amb);
				glMaterialfv(GL_FRONT, GL_DIFFUSE, m_dif);
				glMaterialfv(GL_FRONT, GL_SPECULAR, m_spe);
			
				glNormal3f( 1.0f,  0.0f,  0.0f);
				glVertex3f(-5.0f,  0.0f, -5.0f);

				glNormal3f( 1.0f,  0.0f,  0.0f);
				glVertex3f(-5.0f,  5.0f, -5.0f);

				glNormal3f( 1.0f,  0.0f,  0.0f);
				glVertex3f(-5.0f,  5.0f,  5.0f);

				glNormal3f( 1.0f,  0.0f,  0.0f);
				glVertex3f(-5.0f,  0.0f,  5.0f);
			}
			#pragma endregion

			#pragma region Prawa sciana
			{
				float m_amb[] = { 0.0f, 1.0f, 0.0f, 1.0f };
				float m_dif[] = { 0.0f, 1.0f, 0.0f, 1.0f };
				float m_spe[] = { 0.0f, 0.0f, 0.0f, 1.0f };
				glMaterialfv(GL_FRONT, GL_AMBIENT, m_amb);
				glMaterialfv(GL_FRONT, GL_DIFFUSE, m_dif);
				glMaterialfv(GL_FRONT, GL_SPECULAR, m_spe);
			
				glNormal3f(-1.0f,  0.0f,  0.0f);
				glVertex3f( 5.0f,  5.0f, -5.0f);
		
				glNormal3f(-1.0f,  0.0f,  0.0f);
				glVertex3f( 5.0f,  0.0f, -5.0f);

				glNormal3f(-1.0f,  0.0f,  0.0f);
				glVertex3f( 5.0f,  0.0f,  5.0f);

				glNormal3f(-1.0f,  0.0f,  0.0f);
				glVertex3f( 5.0f,  5.0f,  5.0f);
			}
			#pragma endregion

			#pragma region Tylna sciana
			{
				float m_amb[] = { 1.0f, 1.0f, 1.0f, 1.0f };
				float m_dif[] = { 1.0f, 1.0f, 1.0f, 1.0f };
				float m_spe[] = { 0.0f, 0.0f, 0.0f, 1.0f };
				glMaterialfv(GL_FRONT, GL_AMBIENT, m_amb);
				glMaterialfv(GL_FRONT, GL_DIFFUSE, m_dif);
				glMaterialfv(GL_FRONT, GL_SPECULAR, m_spe);
			
				glNormal3f( 0.0f,  0.0f, -1.0f);
				glVertex3f(-5.0f,  0.0f,  5.0f);

				glNormal3f( 0.0f,  0.0f, -1.0f);
				glVertex3f(-5.0f,  5.0f,  5.0f);

				glNormal3f( 0.0f,  0.0f, -1.0f);
				glVertex3f( 5.0f,  5.0f,  5.0f);

				glNormal3f( 0.0f,  0.0f, -1.0f);
				glVertex3f( 5.0f,  0.0f,  5.0f);
			}
			#pragma endregion

			#pragma region Podloga
			{
				float m_amb[] = { 1.0f, 1.0f, 1.0f, 1.0f };
				float m_dif[] = { 1.0f, 1.0f, 1.0f, 1.0f };
				float m_spe[] = { 0.0f, 0.0f, 0.0f, 1.0f };
				glMaterialfv(GL_FRONT, GL_AMBIENT, m_amb);
				glMaterialfv(GL_FRONT, GL_DIFFUSE, m_dif);
				glMaterialfv(GL_FRONT, GL_SPECULAR, m_spe);
			
				glNormal3f( 0.0f,  1.0f,  0.0f);
				glVertex3f(-5.0f,  0.0f, -5.0f);

				glNormal3f( 0.0f,  1.0f,  0.0f);
				glVertex3f(-5.0f,  0.0f,  5.0f);

				glNormal3f( 0.0f,  1.0f,  0.0f);
				glVertex3f( 5.0f,  0.0f,  5.0f);

				glNormal3f( 0.0f,  1.0f,  0.0f);
				glVertex3f( 5.0f,  0.0f, -5.0f);
			}
			#pragma endregion

			#pragma region Sufit
			{
				float m_amb[] = { 1.0f, 1.0f, 1.0f, 1.0f };
				float m_dif[] = { 1.0f, 1.0f, 1.0f, 1.0f };
				float m_spe[] = { 0.0f, 0.0f, 0.0f, 1.0f };
				glMaterialfv(GL_FRONT, GL_AMBIENT, m_amb);
				glMaterialfv(GL_FRONT, GL_DIFFUSE, m_dif);
				glMaterialfv(GL_FRONT, GL_SPECULAR, m_spe);
			
				glNormal3f( 0.0f, -1.0f,  0.0f);
				glVertex3f(-5.0f,  5.0f,  5.0f);

				glNormal3f( 0.0f, -1.0f,  0.0f);
				glVertex3f(-5.0f,  5.0f, -5.0f);

				glNormal3f( 0.0f, -1.0f,  0.0f);
				glVertex3f( 5.0f,  5.0f, -5.0f);

				glNormal3f( 0.0f, -1.0f,  0.0f);
				glVertex3f( 5.0f,  5.0f,  5.0f);
			}
			#pragma endregion

		glEnd();

	#pragma endregion
	
	#pragma region Rysowanie oteksturowanych quadów

		// W31czamy teksturowanie
		glEnable(GL_TEXTURE_2D);

		// Ustawienie sposobu teksturowania - GL_MODULATE sprawia, ?e owiat3o ma wp3yw na teksture; GL_DECAL i GL_REPLACE rysuj1 teksture tak jak jest
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

		// Ustawienie materia3u
		float m_amb[] = { 0.7f, 0.7f, 0.7f, 1.0f };
		float m_dif[] = { 1.0f, 1.0f, 1.0f, 1.0f };
		float m_spe[] = { 0.0f, 0.0f, 0.0f, 1.0f };
		glMaterialfv(GL_FRONT, GL_AMBIENT, m_amb);
		glMaterialfv(GL_FRONT, GL_DIFFUSE, m_dif);
		glMaterialfv(GL_FRONT, GL_SPECULAR, m_spe);

		#pragma region Skrzynia

			// Wybór tekstury korzystaj1c z jej id
			glBindTexture(GL_TEXTURE_2D, chest_texture);

			glPushMatrix();

			glTranslatef(0.0f, 1.0f, 0.0f);
			glScalef(0.5f, 0.5f, 0.5f);

			int Np = 48;

			// Góra 
			glBegin(GL_QUAD_STRIP);
			float tex_y = 0.75;  // od 0.25 do 0.75
			for (int i = Np; i >= 0; --i) {
				float y = sin(PI * ((float)i / Np));
				float z = cos(PI * ((float)i / Np));
				tex_y -= 0.5 / Np;
				glTexCoord2f(0.75f, tex_y);
				glNormal3f(0.0f, y, z);
				glVertex3f(1.5f, y, z);
				glTexCoord2f(0.25f, tex_y);
				glNormal3f(0.0f, y, z);
				glVertex3f(-1.0f, y, z);
			}
			glEnd();

			// Góra prawego boku
			glBegin(GL_TRIANGLE_FAN);
			float text_x1 = 0.35f;
			float text_y1 = 0.55f;
			glTexCoord2f(0.55f, 0.5f);
			glNormal3f(1.0f, 0.0f, 0.0f);
			glVertex3f(1.0f, 0.0f, 0.0f);
			for (int i = Np; i >= 0; --i) {
				float y = sin(PI * ((float)i / Np));
				float z = cos(PI * ((float)i / Np));
				text_x1 += 0.15 * z;
				text_y1 += 0.15 * y;
				glTexCoord2f(text_x1, text_y1);
				glNormal3f(1.5f, 0.0f, 0.0f);
				glVertex3f(1.5f, y, z);
			}
			glEnd();

			// Góra lewego boku
			glBegin(GL_TRIANGLE_FAN);
			float text_x = 0.35f;
			float text_y = 0.55f;
			glTexCoord2f(0.5f, 0.55f);
			glNormal3f(-1.0f, 0.0f, 0.0f);
			glVertex3f(-1.0f, 0.0f, 0.0f);
			for (int i = 0; i <= Np; ++i) {
				float y = sin(PI * ((float)i / Np));
				float z = cos(PI * ((float)i / Np));
				text_x = 0.15 * z;
				text_y = 0.15 * y;
				glTexCoord2f(text_x, text_y);
				glNormal3f(-1.0f, 0.0f, 0.0f);
				glVertex3f(-1.0f, y, z);
			}
			glEnd();

			glBegin(GL_QUADS);

			// Przod
			glTexCoord2f(0.75f, 1.0f);
			glNormal3f(0.0f, 0.0f, 1.0f);
			glVertex3f(1.5f, 0.0f, 1.0f);

			glTexCoord2f(0.25f, 1.0f);
			glNormal3f(0.0f, 0.0f, 1.0f);
			glVertex3f(-1.0f, 0.0f, 1.0f);

			glTexCoord2f(0.25f, 0.8f);
			glNormal3f(0.0f, 0.0f, 1.0f);
			glVertex3f(-1.0f, -1.0f, 1.0f);

			glTexCoord2f(0.75f, 0.8f);
			glNormal3f(0.0f, 0.0f, 1.0f);
			glVertex3f(1.5f, -1.0f, 1.0f);

			// Tyl
			glTexCoord2f(0.25f, 0.75f);
			glNormal3f(0.0f, 0.0f, -1.0f);
			glVertex3f(-1.0f, 0.0f, -1.0f);

			glTexCoord2f(0.75f, 0.75f);
			glNormal3f(0.0f, 0.0f, -1.0f);
			glVertex3f(1.5f, 0.0f, -1.0f);

			glTexCoord2f(0.75f, 0.55f);
			glNormal3f(0.0f, 0.0f, -1.0f);
			glVertex3f(1.5f, -1.0f, -1.0f);

			glTexCoord2f(0.25f, 0.55f);
			glNormal3f(0.0f, 0.0f, -1.0f);
			glVertex3f(-1.0f, -1.0f, -1.0f);

			// Lewa
			glTexCoord2f(0.65f, 0.55f);
			glNormal3f(-1.0f, 0.0f, 0.0f);
			glVertex3f(-1.0f, 0.0f, 1.0f);

			glTexCoord2f(0.35f, 0.55f);
			glNormal3f(-1.0f, 0.0f, 0.0f);
			glVertex3f(-1.0f, 0.0f, -1.0f);

			glTexCoord2f(0.35f, 0.35f);
			glNormal3f(-1.0f, 0.0f, 0.0f);
			glVertex3f(-1.0f, -1.0f, -1.0f);

			glTexCoord2f(0.65f, 0.35f);
			glNormal3f(-1.0f, 0.0f, 0.0f);
			glVertex3f(-1.0f, -1.0f, 1.0f);

			// Prawa
			glTexCoord2f(0.35f, 0.55f);
			glNormal3f(1.0f, 0.0f, 0.0f);
			glVertex3f(1.5f, 0.0f, -1.0f);

			glTexCoord2f(0.65f, 0.55f);
			glNormal3f(1.0f, 0.0f, 0.0f);
			glVertex3f(1.5f, 0.0f, 1.0f);

			glTexCoord2f(0.65f, 0.35f);
			glNormal3f(1.0f, 0.0f, 0.0f);
			glVertex3f(1.5f, -1.0f, 1.0f);

			glTexCoord2f(0.35f, 0.35f);
			glNormal3f(1.0f, 0.0f, 0.0f);
			glVertex3f(1.5f, -1.0f, -1.0f);

			// Dol
			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(1.0f, -1.0f, -1.0f);

			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(1.0f, -1.0f, 1.0f);

			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(-1.0f, -1.0f, 1.0f);

			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(-1.0f, -1.0f, -1.0f);

			glEnd();

			glPopMatrix();

		#pragma endregion

		// Wy31czenie teksturowania - geometria renderowana od tego miejsca nie bedzie ju? oteksturowana
		glDisable(GL_TEXTURE_2D);

	#pragma endregion

	glutSwapBuffers();
	glFlush();
	glutPostRedisplay();

}

void OnReshape(int width, int height) {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, width, height);
	gluPerspective(60.0f, (float) width / height, .01f, 100.0f);
}