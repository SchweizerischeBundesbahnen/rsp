// ESTA-Community-Helper for Python (https://issues.sbb.ch/browse/ESTA-3822): Status alpha!
// https://code.sbb.ch/projects/KD_ESTA_BLUEPRINTS/repos/esta-python-helper/browse
// https://code.sbb.ch/projects/KD_ESTA_BLUEPRINTS/repos/esta-python-lib/browse/Jenkinsfile
library identifier: 'python-helper@master',
        retriever: modernSCM(
                [$class       : 'GitSCMSource',
                 credentialsId: 'fsosebuild',
                 remote       : 'https://code.sbb.ch/scm/KD_ESTA_BLUEPRINTS/esta-python-helper.git'])

pipeline {
    // aws label required, no access to internet from default vias nodes: https://issues.sbb.ch/servicedesk/customer/portal/1/CLEW-895
    agent { label 'aws' }
    options {
        timestamps()
        disableConcurrentBuilds()
        // https://stackoverflow.com/questions/39542485/how-to-write-pipeline-to-discard-old-builds
        // maximal 5 Builds und maximal 30 Tage
        buildDiscarder(logRotator(numToKeepStr: '5', artifactNumToKeepStr: '5', daysToKeepStr: '30', artifactDaysToKeepStr: '30'))
    }
    environment {
        // access token with repo:status permission created via https://github.com/settings/tokens in personal account
        // uploaded to Jenkins via https://ssp.app.ose.sbb-cloud.net/wzu/jenkinscredentials
        // list of credentials: https://ci.sbb.ch/job/KS_PFI/credentials/
        GITHUB_TOKEN = credentials('19fbbc7a-7243-431c-85d8-0f1cc63d413b')
    }
    stages {
        stage('github pending') {

            steps {
                // https://developer.github.com/v3/repos/statuses/
                script {
                    sh """
echo $GITHUB_TOKEN > token.txt
curl --insecure -v --request POST -H "Authorization: token ${
                        GITHUB_TOKEN
                    }" https://api.github.com/repos/SchweizerischeBundesbahnen/rsp/statuses/${
                        GIT_COMMIT
                    } --data '{ "state": "pending", "target_url": "'${BUILD_URL}'", "description": "The build has started!", "context": "continuous-integration/jenkins" }'
"""
                }
            }
        }
        stage('test') {
            steps {
                tox_conda_wrapper(
                        ENVIRONMENT_YAML: 'rsp_environment.yml',
                        WHAT: {
                            sh """
conda config --show channels
python -m tox . --recreate -v
"""
                        })
            }
        }
    }
    post {
        failure {
            // https://developer.github.com/v3/repos/statuses/
            sh """
curl --insecure -v --request POST -H "Authorization: token ${
                GITHUB_TOKEN
            }" https://api.github.com/repos/SchweizerischeBundesbahnen/rsp/statuses/${
                GIT_COMMIT
            } --data '{ "state": "failure", "target_url": "'${BUILD_URL}'", "description": "The build has failed!", "context": "continuous-integration/jenkins" }'
"""
        }
        success {
            // https://developer.github.com/v3/repos/statuses/
            sh """
curl --insecure -v --request POST -H "Authorization: token ${
                GITHUB_TOKEN
            }" https://api.github.com/repos/SchweizerischeBundesbahnen/rsp/statuses/${
                GIT_COMMIT
            } --data '{ "state": "success", "target_url": "'${BUILD_URL}'", "description": "The build has succeeded!", "context": "continuous-integration/jenkins" }'
"""
        }
    }
}



