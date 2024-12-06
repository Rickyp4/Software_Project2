pipeline {
	 agent any
	 stages {
	 	stage('Hello'){
			steps {
			      echo 'Hello'
			      }
		}
		stage('Testing'){
			steps {
			      echo 'No testing in current stage'
			      }
		}
	}
	post {
	     always {
		    }
		    
    	     }
}


