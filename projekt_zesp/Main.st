PROGRAM _INIT
    //check water level from sensor
    //set water level from sensor
    //set pumpPower =0
    (* Insert code here *)
	Tank.aoPump :=0;
	Tank.uiLevel := INT_TO_REAL(Tank.aiLevel);
	Unit.uiLevel := 1;
   
     
END_PROGRAM
 
PROGRAM _CYCLIC
    Tank.uiLevel := INT_TO_REAL(Tank.aiLevel);
    uiStatus:=1;
	CASE uiStatus OF
		0:
			Tank.aoPump :=0;
		1:
			CASE uiControlType OF
			0:
			Tank.aoPump :=0;		
			1:
				//uzycie
	       
				IF Tank.aiLevel < TankPar.uiLevelHi AND Tank.aiLevel > TankPar.uiLevelLo THEN
					IF (REAL_TO_INT(Tank.uiLevel) - Tank.aiLevel) > 100 THEN
						Tank.aoPump := (REAL_TO_INT(Tank.uiLevel) - Tank.aiLevel)*10;
					END_IF;
				ELSE
					Tank.aoPump :=0;
				END_IF;
	   
				IF Tank.aiLevel < TankPar.uiLevelLoLo THEN
					Tank.aoPump := TankPar.uiMaxPumpLevel;
				END_IF;
	   
				IF Tank.aiLevel > TankPar.uiLevelHiHi THEN
					Tank.aoPump := -TankPar.uiMaxPumpLevel;
				END_IF;
	           
			2: // serwis
				Tank.aoPump := REAL_TO_INT(Tank.uiPump);
	 
			END_CASE;
	END_CASE;
	
	
END_PROGRAM
 
PROGRAM _EXIT
	Tank.aoPump :=0;
     
END_PROGRAM